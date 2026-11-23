from __future__ import annotations

import asyncio
import datetime as dt
import json
import secrets
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from mlops_shared.contract import RunDir, read_eval_results, read_export_manifest, read_metrics, read_status
from mlops_shared.recipes import RecipeSpec, commercial_ok, parse_dataset_ref

from .. import system
from ..backends import load_capabilities
from ..db import get_session
from ..models import Dataset, DatasetVersion, Recipe, RecipeVersion, Run
from .datasets import iso
from .recipes import check_recipe

router = APIRouter(prefix="/api")

EVENTS_POLL_S = 1.0
TERMINAL = ("done", "failed", "cancelled")
OOM_HINT = "CUDA out of memory: lower batch_size or enable gradient checkpointing"


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
    errors = await check_recipe(session, spec, load_capabilities(settings.backends_root))
    if errors:
        return JSONResponse({"errors": errors}, status_code=422)
    train = await _dataset_version(session, spec.datasets.train)
    hashes = {"train": train.hash}
    if spec.datasets.eval:
        hashes["eval"] = (await _dataset_version(session, spec.datasets.eval)).hash
    rv.frozen = True
    run_id = f"r-{secrets.token_hex(4)}"
    session.add(Run(
        id=run_id, recipe_version_id=rv.id, state="queued", run_dir=str(settings.runs_dir / run_id),
        commercial_ok=commercial_ok(spec.base_model_meta, [bool((train.provenance or {}).get("commercial_ok"))]),
        dataset_hashes=hashes,
    ))
    await session.commit()
    return {"id": run_id}


@router.get("/runs")
async def list_runs(state: str | None = None, session: AsyncSession = Depends(get_session)) -> list[dict]:
    q = (select(Run, Recipe.name).join(RecipeVersion, RecipeVersion.id == Run.recipe_version_id)
         .join(Recipe, Recipe.id == RecipeVersion.recipe_id).order_by(Run.created_at.desc(), Run.id.desc()))
    if state:
        q = q.where(Run.state == state)
    return [
        {"id": r.id, "recipe": name, "state": r.state, "commercial_ok": r.commercial_ok,
         "created_at": iso(r.created_at), "started_at": iso(r.started_at), "finished_at": iso(r.finished_at)}
        for r, name in (await session.execute(q)).all()
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
        "mlflow_run_id": run.mlflow_run_id, "created_at": iso(run.created_at),
        "started_at": iso(run.started_at), "finished_at": iso(run.finished_at),
        "recipe": {"name": recipe.name, "number": rv.number},
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


def _tail(path: Path, n: int) -> list[str]:
    try:
        text = path.read_bytes().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return []
    return text.splitlines()[-n:] if n > 0 else []


@router.get("/runs/{run_id}/logs")
async def run_logs(run_id: str, tail: int = 200, session: AsyncSession = Depends(get_session)) -> dict:
    run = await get_run(session, run_id)
    logs = RunDir(Path(run.run_dir)).logs_dir
    return {"stdout": _tail(logs / "stdout.log", tail), "stderr": _tail(logs / "stderr.log", tail)}


def _new_lines(path: Path, offset: int, final: bool = False) -> tuple[list[str], int]:
    """Complete lines appended after byte `offset`; `final` also returns an unterminated last line."""
    try:
        with path.open("rb") as f:
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
    log_off = {"stdout": 0, "stderr": 0}
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


@router.get("/runs/{run_id}/files/{path:path}")
async def run_file(run_id: str, path: str, session: AsyncSession = Depends(get_session)) -> Response:
    run = await get_run(session, run_id)
    rd = RunDir(Path(run.run_dir))
    target = Path(run.run_dir) / path
    if not inside(target, rd.eval_dir, rd.export_dir):
        raise HTTPException(404, "file not found")
    return FileResponse(target.resolve())
