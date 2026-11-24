from __future__ import annotations

import asyncio
import datetime as dt
import json
import re
import os
import secrets
import zipfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.core.contract import (
    RunDir, read_eval_results, read_eval_samples, read_export_manifest, read_metrics, read_prepared_refs, read_status,
)
from app.core.recipes import RecipeSpec, commercial_ok, parse_dataset_ref

from .. import system
from ..backends import load_capabilities
from ..db import get_session
from ..runs_init import list_checkpoints, resolve_init
from ..models import Dataset, DatasetVersion, Recipe, RecipeVersion, Run
from .datasets import iso
from .recipes import check_recipe

router = APIRouter(prefix="/api")

EVENTS_POLL_S = 1.0
TERMINAL = ("done", "failed", "cancelled")
OOM_HINT = "CUDA out of memory: lower batch_size or enable gradient checkpointing"
AUDIO_NAME = re.compile(r"^[A-Za-z0-9._-]+\.wav$")
LOG_WINDOW_BYTES = 256 * 1024  # first read of a log only looks at its end


class CreateRun(BaseModel):
    recipe: str
    version: int | None = None


async def get_run(session: AsyncSession, run_id: str) -> Run:
    run = await session.get(Run, run_id)
    if run is None:
        raise HTTPException(404, f"run '{run_id}' not found")
    return run


async def run_recipe(session: AsyncSession, run: Run) -> tuple[Recipe, RecipeVersion]:
    rv = await session.get(RecipeVersion, run.recipe_version_id)
    return await session.get(Recipe, rv.recipe_id), rv


def inside(path: Path, *roots: Path) -> bool:
    """True if `path` resolves to an existing file under one of `roots`."""
    resolved = path.resolve()
    return resolved.is_file() and any(root.resolve() in resolved.parents for root in roots)


def inside_dir(path: Path, root: Path) -> bool:
    """True if `path` resolves to an existing directory strictly under `root`."""
    resolved = path.resolve()
    return resolved.is_dir() and root.resolve() in resolved.parents


ZIP_CHUNK = 1024 * 1024


class _ChunkSink:
    """Write-only, non-seekable file object: zipfile writes into it, the generator drains it."""

    def __init__(self) -> None:
        self.buf = bytearray()
        self.pos = 0

    def write(self, b) -> int:
        self.buf += b
        self.pos += len(b)
        return len(b)

    def tell(self) -> int:
        return self.pos

    def flush(self) -> None:
        pass

    def drain(self) -> bytes:
        out = bytes(self.buf)
        self.buf.clear()
        return out


def _dir_files(src: Path) -> Iterator[tuple[Path, str]]:
    """(file, arcname) for every regular file under `src` whose real path stays inside `src`."""
    root = src.resolve()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for name in sorted(filenames):
            f = Path(dirpath) / name
            real = f.resolve()
            if root in real.parents and real.is_file():
                yield real, f"{src.name}/{f.relative_to(root).as_posix()}"


def _zip_stream(src: Path) -> Iterator[bytes]:
    """Stream `src` as a ZIP_STORED archive chunk by chunk; nothing is buffered beyond one chunk."""
    sink = _ChunkSink()
    with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for real, arcname in _dir_files(src):
            info = zipfile.ZipInfo.from_file(real, arcname)
            info.compress_type = zipfile.ZIP_STORED
            with real.open("rb") as fin, zf.open(info, "w", force_zip64=True) as fout:
                while chunk := fin.read(ZIP_CHUNK):
                    fout.write(chunk)
                    if len(sink.buf) >= ZIP_CHUNK:
                        yield sink.drain()
            yield sink.drain()
    yield sink.drain()


def artifact_response(target: Path, file_roots: tuple[Path, ...], dir_root: Path | None) -> Response:
    """Serve a file under `file_roots`, or a directory under `dir_root` as a streamed `<name>.zip`."""
    if inside(target, *file_roots):
        return FileResponse(target.resolve())
    if dir_root is not None and inside_dir(target, dir_root):
        real = target.resolve()
        return StreamingResponse(
            _zip_stream(real), media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{real.name}.zip"'})
    raise HTTPException(404, "file not found")


def _check_disk(request: Request) -> None:
    if system.disk_status(request.app.state.settings).block:
        raise HTTPException(507, "not enough free disk space to start a run")


async def _dataset_version(session: AsyncSession, ref: str) -> DatasetVersion:
    name, number = parse_dataset_ref(ref)
    q = (select(DatasetVersion).join(Dataset, Dataset.id == DatasetVersion.dataset_id)
         .where(Dataset.name == name, DatasetVersion.number == number))
    return (await session.execute(q)).scalar_one()


@router.post("/runs", status_code=201, response_model=None)
async def create_run(body: CreateRun, request: Request, session: AsyncSession = Depends(get_session)):
    settings = request.app.state.settings
    recipe = (await session.execute(select(Recipe).where(Recipe.name == body.recipe))).scalar_one_or_none()
    if recipe is None:
        raise HTTPException(404, f"recipe '{body.recipe}' not found")
    q = select(RecipeVersion).where(RecipeVersion.recipe_id == recipe.id)
    q = q.where(RecipeVersion.number == body.version) if body.version else q.order_by(RecipeVersion.number.desc())
    rv = (await session.execute(q.limit(1))).scalar_one_or_none()
    if rv is None:
        raise HTTPException(404, f"version {body.version} of '{body.recipe}' not found")
    _check_disk(request)
    spec = RecipeSpec.model_validate(rv.spec)
    errors = await check_recipe(session, spec, load_capabilities(settings.backends_root), settings)
    if errors:
        return JSONResponse({"errors": errors}, status_code=422)
    train = await _dataset_version(session, spec.datasets.train)
    hashes = {"train": train.hash}
    if spec.datasets.eval:
        hashes["eval"] = (await _dataset_version(session, spec.datasets.eval)).hash
    resolved, init_errors = await resolve_init(session, spec, settings)
    if init_errors:
        return JSONResponse({"errors": init_errors}, status_code=422)
    parent, _, init_ref = resolved if resolved else (None, None, None)
    rv.frozen = True
    run_id = f"r-{secrets.token_hex(4)}"
    session.add(Run(
        id=run_id, recipe_version_id=rv.id, state="queued", run_dir=str(settings.runs_dir / run_id),
        commercial_ok=commercial_ok(
            spec.base_model_meta, [bool((train.provenance or {}).get("commercial_ok"))]
            + ([parent.commercial_ok] if parent else [])),
        dataset_hashes=hashes,
        parent_run_id=parent.id if parent else None,
        parent_checkpoint=init_ref.checkpoint if init_ref else None,
    ))
    await session.commit()
    return {"id": run_id}


@router.get("/runs")
async def list_runs(state: str | None = None, limit: int | None = Query(None, ge=1, le=500),
                    session: AsyncSession = Depends(get_session)) -> list[dict]:
    q = (select(Run, Recipe.name, RecipeVersion).join(RecipeVersion, RecipeVersion.id == Run.recipe_version_id)
         .join(Recipe, Recipe.id == RecipeVersion.recipe_id).order_by(Run.created_at.desc(), Run.id.desc()))
    if state:
        q = q.where(Run.state == state)
    if limit is not None:
        q = q.limit(limit)
    return [
        {"id": r.id, "recipe": name, "recipe_number": rv.number, "backend": rv.spec.get("backend"), "task": rv.spec.get("task"), "state": r.state, "commercial_ok": r.commercial_ok,
         "created_at": iso(r.created_at), "started_at": iso(r.started_at), "finished_at": iso(r.finished_at)}
        for r, name, rv in (await session.execute(q)).all()
    ]


@router.get("/runs/{run_id}")
async def run_detail(run_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    run = await get_run(session, run_id)
    recipe, rv = await run_recipe(session, run)
    rd = RunDir(Path(run.run_dir))
    status = read_status(rd)
    results = read_eval_results(rd)
    manifest = read_export_manifest(rd)
    return {
        "id": run.id, "state": run.state, "attempt": run.attempt, "resume": run.resume,
        "cancel_requested": run.cancel_requested, "run_dir": run.run_dir,
        "commercial_ok": run.commercial_ok, "error": run.error, "oom": run.oom, "git_sha": run.git_sha,
        "lock_hash": run.lock_hash, "dataset_hashes": run.dataset_hashes,
        "mlflow_run_id": run.mlflow_run_id, "parent_run_id": run.parent_run_id,
        "parent_checkpoint": run.parent_checkpoint, "created_at": iso(run.created_at),
        "started_at": iso(run.started_at), "finished_at": iso(run.finished_at),
        "recipe": {"name": recipe.name, "number": rv.number}, "task": rv.spec.get("task"),
        "status": status.model_dump(mode="json") if status else None,
        "eval": results.model_dump(mode="json") if results else None,
        "exports": [a.model_dump(mode="json") for a in manifest.artifacts] if manifest else [],
        "oom_hint": OOM_HINT if run.oom else None,
    }


@router.post("/runs/{run_id}/cancel", response_model=None)
async def cancel_run(run_id: str, session: AsyncSession = Depends(get_session)):
    run = await get_run(session, run_id)
    if run.state in TERMINAL:
        raise HTTPException(409, f"run is already {run.state}")
    if run.state == "queued":
        # conditional: the worker may claim the run (queued -> preparing) between our read and write
        res = await session.execute(
            update(Run).where(Run.id == run_id, Run.state == "queued")
            .values(state="cancelled", finished_at=dt.datetime.now(dt.timezone.utc)))
        await session.commit()
        if res.rowcount:
            return {"state": "cancelled"}
        await session.refresh(run)
        if run.state in TERMINAL:
            raise HTTPException(409, f"run is already {run.state}")
    run.cancel_requested = True
    await session.commit()
    return JSONResponse({"state": run.state, "cancel_requested": True}, status_code=202)


@router.post("/runs/{run_id}/resume")
async def resume_run(run_id: str, request: Request, session: AsyncSession = Depends(get_session)) -> dict:
    run = await get_run(session, run_id)
    if run.state not in ("failed", "cancelled"):
        raise HTTPException(409, f"cannot resume a run in state {run.state}")
    _check_disk(request)
    run.state, run.resume, run.attempt = "queued", True, run.attempt + 1
    run.cancel_requested, run.error, run.oom = False, None, False
    run.finished_at = None
    await session.commit()
    return {"state": run.state, "attempt": run.attempt}


@router.get("/runs/{run_id}/metrics")
async def run_metrics(run_id: str, offset: int = 0, session: AsyncSession = Depends(get_session)) -> dict:
    run = await get_run(session, run_id)
    items, next_offset = read_metrics(RunDir(Path(run.run_dir)), max(0, offset))
    return {"items": [m.model_dump(mode="json") for m in items], "next_offset": next_offset}


def _tail_start(f, size: int) -> int:
    """Byte offset of the first complete line within the last LOG_WINDOW_BYTES of an open file."""
    start = max(0, size - LOG_WINDOW_BYTES)
    if start == 0:
        return 0
    f.seek(start - 1)
    if f.read(1) == b"\n":  # the window starts exactly at a line boundary
        return start
    nl = f.read(LOG_WINDOW_BYTES).find(b"\n")
    return size if nl < 0 else start + nl + 1


def _tail(path: Path, n: int) -> list[str]:
    if n <= 0:
        return []
    try:
        with path.open("rb") as f:
            size = f.seek(0, 2)
            f.seek(_tail_start(f, size))
            text = f.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return []
    return text.splitlines()[-n:]


@router.get("/runs/{run_id}/checkpoints")
async def run_checkpoints(run_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    run = await get_run(session, run_id)
    return {"items": list_checkpoints(RunDir(Path(run.run_dir)))}


@router.get("/runs/{run_id}/logs")
async def run_logs(run_id: str, tail: int = 200, session: AsyncSession = Depends(get_session)) -> dict:
    run = await get_run(session, run_id)
    logs = RunDir(Path(run.run_dir)).logs_dir
    return {"stdout": _tail(logs / "stdout.log", tail), "stderr": _tail(logs / "stderr.log", tail)}


def _new_lines(path: Path, offset: int | None, final: bool = False) -> tuple[list[str], int | None]:
    """Complete lines appended after byte `offset`; `final` also returns an unterminated last line.

    `offset=None` is the first read: start at the first whole line of the last LOG_WINDOW_BYTES."""
    try:
        with path.open("rb") as f:
            if offset is None:
                offset = _tail_start(f, f.seek(0, 2))
            f.seek(offset)
            data = f.read()
    except FileNotFoundError:
        return [], offset
    end = len(data) if final else data.rfind(b"\n") + 1
    lines = data[:end].decode("utf-8", errors="replace").splitlines()
    return lines, offset + end


def _event(name: str, data: dict) -> dict:
    return {"event": name, "data": json.dumps(data)}


async def _stream(request: Request, run_id: str) -> AsyncIterator[dict]:
    sessionmaker = request.app.state.sessionmaker
    last_state: str | None = None
    last_status: str | None = None
    metric_off = 0
    log_off: dict[str, int | None] = {"stdout": None, "stderr": None}  # None: first read (tail window)
    while True:
        async with sessionmaker() as session:
            run = await session.get(Run, run_id)
        if run is None:
            return
        terminal = run.state in TERMINAL
        if run.state != last_state and not terminal:
            yield _event("state", {"state": run.state})
        last_state = run.state
        rd = RunDir(Path(run.run_dir))
        status = read_status(rd)
        dumped = status.model_dump_json() if status else None
        if dumped is not None and dumped != last_status:
            last_status = dumped
            yield _event("status", json.loads(dumped))
        metrics, metric_off = read_metrics(rd, metric_off)
        for m in metrics:
            yield _event("metric", m.model_dump(mode="json"))
        for stream in ("stdout", "stderr"):
            lines, log_off[stream] = _new_lines(rd.logs_dir / f"{stream}.log", log_off[stream], terminal)
            for line in lines:
                yield _event("log", {"stream": stream, "line": line})
        if terminal:
            yield _event("state", {"state": run.state})
            return
        await asyncio.sleep(EVENTS_POLL_S)


@router.get("/runs/{run_id}/events")
async def run_events(run_id: str, request: Request):
    # own short-lived session: a dependency session would stay open (pinning a pooled connection)
    # for the whole stream
    async with request.app.state.sessionmaker() as session:
        await get_run(session, run_id)
    return EventSourceResponse(_stream(request, run_id))


@router.get("/runs/{run_id}/prepared")
async def run_prepared(run_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    run = await get_run(session, run_id)
    return {"refs": read_prepared_refs(RunDir(Path(run.run_dir)))}


@router.get("/runs/{run_id}/eval/samples")
async def run_eval_samples(run_id: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=500),
                           session: AsyncSession = Depends(get_session)) -> dict:
    run = await get_run(session, run_id)
    items, total = read_eval_samples(RunDir(Path(run.run_dir)), offset, limit)
    return {"items": items, "total": total}


@router.get("/runs/{run_id}/eval/audio/{name}")
async def run_eval_audio(run_id: str, name: str, session: AsyncSession = Depends(get_session)) -> Response:
    run = await get_run(session, run_id)
    audio_dir = RunDir(Path(run.run_dir)).eval_audio_dir
    if not AUDIO_NAME.match(name):
        raise HTTPException(404, "file not found")
    target = (audio_dir / name).resolve()
    if not inside(target, audio_dir.resolve()):
        raise HTTPException(404, "file not found")
    return FileResponse(target, media_type="audio/wav")


@router.get("/runs/{run_id}/files/{path:path}")
async def run_file(run_id: str, path: str, session: AsyncSession = Depends(get_session)) -> Response:
    run = await get_run(session, run_id)
    rd = RunDir(Path(run.run_dir))
    target = Path(run.run_dir) / path
    return artifact_response(target, (rd.eval_dir, rd.export_dir), rd.export_dir)
