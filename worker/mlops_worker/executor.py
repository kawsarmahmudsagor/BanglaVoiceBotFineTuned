"""Run executor: turns a claimed Run into config.json and drives the backend subprocess per phase."""

import asyncio
import hashlib
import logging
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Protocol

from sqlalchemy import select, update

from mlops_api.backends import backend_python
from mlops_api.models import Dataset, DatasetVersion, RecipeVersion, Run, utcnow
from mlops_shared.contract import (
    DatasetRef, Metric, RunConfig, RunDir, read_metrics, write_config,
)
from mlops_shared.recipes import RecipeSpec, parse_dataset_ref

from .lock import release

log = logging.getLogger(__name__)

POLL_S = 1.0
PHASES = ("prepare", "train", "eval", "export")
PHASE_STATE = {"prepare": "preparing", "train": "training", "eval": "evaluating", "export": "exporting"}
_OOM_RE = re.compile(r"CUDA out of memory|OutOfMemoryError")
_LOG_TAIL_BYTES = 1_000_000


class Forwarder(Protocol):
    def start(self, run_id: str, params: dict, tags: dict) -> str | None: ...
    def metrics(self, items: list[Metric]) -> None: ...
    def finish(self, state: str) -> None: ...


class NullForwarder:
    def start(self, run_id: str, params: dict, tags: dict) -> str | None:
        return None

    def metrics(self, items: list[Metric]) -> None:
        return None

    def finish(self, state: str) -> None:
        return None


def _ref(version: DatasetVersion, name: str, splits: list[str]) -> DatasetRef:
    return DatasetRef(
        name=name,
        version=version.number,
        hash=version.hash,
        root=version.path,
        manifest=str(Path(version.path) / "manifest.parquet"),
        splits=splits,
    )


def build_config(
    run: Run,
    recipe: RecipeSpec,
    versions: dict[str, DatasetVersion],
    settings,
    *,
    eval_only: set[str] | frozenset[str],
) -> RunConfig:
    """`versions` maps recipe refs (`name@vN`) to versions; `eval_only` holds eval-only dataset names."""
    datasets: dict[str, DatasetRef] = {}
    name, _ = parse_dataset_ref(recipe.datasets.train)
    datasets["train"] = _ref(versions[recipe.datasets.train], name, ["train", "val"])
    if recipe.datasets.eval:
        name, _ = parse_dataset_ref(recipe.datasets.eval)
        splits = ["train", "val", "test"] if name in eval_only else ["test"]
        datasets["eval"] = _ref(versions[recipe.datasets.eval], name, splits)
    return RunConfig(
        run_id=run.id,
        attempt=run.attempt,
        resume=run.resume,
        task=recipe.task,
        backend=recipe.backend,
        base_model=recipe.base_model,
        method=recipe.method,
        hparams=recipe.hparams,
        export=recipe.export,
        seed=recipe.seed,
        datasets=datasets,
        prepared_root=str(settings.prepared_dir),
    )


async def _update_run(session_factory, run_id: str, **values) -> None:
    async with session_factory() as s:
        await s.execute(update(Run).where(Run.id == run_id).values(**values))
        await s.commit()


async def _cancel_requested(session_factory, run_id: str) -> bool:
    async with session_factory() as s:
        return bool((await s.execute(select(Run.cancel_requested).where(Run.id == run_id))).scalar_one())


def _git_sha(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _lock_hash(settings, backend: str) -> str | None:
    p = Path(settings.backends_root) / backend / "uv.lock"
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def _safe_forward(fn, *args):
    try:
        return fn(*args)
    except Exception:
        log.warning("forwarder call failed", exc_info=True)
        return None


def _drain_metrics(rd: RunDir, offset: int, forwarder: Forwarder) -> int:
    items, offset = read_metrics(rd, offset)
    if items:
        _safe_forward(forwarder.metrics, items)
    return offset


def _read_from(path: Path, start: int) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            f.seek(max(start, size - _LOG_TAIL_BYTES))
            return f.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _spawn_kwargs() -> dict:
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _signal(proc: asyncio.subprocess.Process, sig: int, win_method: str) -> None:
    """Signal the backend's whole process group (POSIX) or the process (Windows)."""
    try:
        if sys.platform == "win32":
            getattr(proc, win_method)()
        else:
            os.killpg(proc.pid, sig)  # session leader: pgid == pid
    except (ProcessLookupError, PermissionError):
        pass


async def _stop(proc: asyncio.subprocess.Process, grace_s: float) -> None:
    if proc.returncode is None:
        _signal(proc, signal.SIGTERM, "terminate")
        try:
            await asyncio.wait_for(proc.wait(), grace_s)
        except asyncio.TimeoutError:
            _signal(proc, signal.SIGKILL if sys.platform != "win32" else signal.SIGTERM, "kill")
    await proc.wait()


async def _run_phase(session_factory, run_id, settings, python: Path, rd: RunDir, phase: str,
                     forwarder: Forwarder, offset: int) -> tuple[int, int, bool]:
    """Returns (exit_code, metrics_offset, cancelled)."""
    rd.logs_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    cancelled = False
    with (rd.logs_dir / "stdout.log").open("ab") as out, (rd.logs_dir / "stderr.log").open("ab") as err:
        proc = await asyncio.create_subprocess_exec(
            str(python), "-m", "mlops_backend", phase, "--run-dir", str(rd.path),
            stdout=out, stderr=err, env=env, **_spawn_kwargs(),
        )
        try:
            while proc.returncode is None:
                try:
                    await asyncio.wait_for(proc.wait(), POLL_S)
                except asyncio.TimeoutError:
                    pass
                offset = _drain_metrics(rd, offset, forwarder)
                if proc.returncode is None and await _cancel_requested(session_factory, run_id):
                    cancelled = True
                    await _stop(proc, settings.cancel_grace_s)
        finally:
            if proc.returncode is None:
                _signal(proc, signal.SIGKILL if sys.platform != "win32" else signal.SIGTERM, "kill")
                await proc.wait()
    offset = _drain_metrics(rd, offset, forwarder)
    return proc.returncode, offset, cancelled


async def _load(session_factory, run_id: str):
    """Load the run, its recipe spec and dataset versions."""
    async with session_factory() as s:
        run = (await s.execute(select(Run).where(Run.id == run_id).execution_options(populate_existing=True))).scalar_one()
        rv = await s.get(RecipeVersion, run.recipe_version_id)
        spec = RecipeSpec.model_validate(rv.spec)
        versions: dict[str, DatasetVersion] = {}
        eval_only: set[str] = set()
        for ref in [spec.datasets.train, spec.datasets.eval]:
            if not ref:
                continue
            name, number = parse_dataset_ref(ref)
            ds = (await s.execute(select(Dataset).where(Dataset.name == name))).scalar_one_or_none()
            ver = None
            if ds is not None:
                ver = (await s.execute(
                    select(DatasetVersion).where(DatasetVersion.dataset_id == ds.id, DatasetVersion.number == number)
                )).scalar_one_or_none()
            if ver is None:
                raise ValueError(f"dataset {ref} not found")
            versions[ref] = ver
            if ds.eval_only:
                eval_only.add(name)
        s.expunge_all()
        return run, spec, versions, eval_only


async def execute_run(session_factory, run_id: str, settings, forwarder: Forwarder) -> str:
    """Execute a claimed run to a terminal state. Never raises; always releases the GPU lock."""
    state = "failed"
    try:
        state = await _execute(session_factory, run_id, settings, forwarder)
    except asyncio.CancelledError:
        state = "failed"
        try:
            await _update_run(session_factory, run_id, error="worker interrupted")
        except Exception:
            log.exception("could not record interruption for run %s", run_id)
        raise
    except Exception as exc:
        log.exception("run %s crashed", run_id)
        state = "failed"
        try:
            await _update_run(session_factory, run_id, error=f"{type(exc).__name__}: {exc}")
        except Exception:
            log.exception("could not record failure for run %s", run_id)
    finally:
        try:
            await _update_run(session_factory, run_id, state=state, finished_at=utcnow())
        except Exception:
            log.exception("could not set final state for run %s", run_id)
        try:
            async with session_factory() as s:
                await release(s, "run", run_id)
        except Exception:
            log.exception("could not release GPU lock for run %s", run_id)
        _safe_forward(forwarder.finish, state)
    return state


async def _execute(session_factory, run_id: str, settings, forwarder: Forwarder) -> str:
    run, spec, versions, eval_only = await _load(session_factory, run_id)
    python = backend_python(spec.backend, settings)  # BackendMissing -> failed run with message
    rd = RunDir(Path(run.run_dir))
    rd.ensure()
    cfg = build_config(run, spec, versions, settings, eval_only=eval_only)
    write_config(rd, cfg)

    values: dict = {
        "git_sha": await asyncio.to_thread(_git_sha, Path(settings.backends_root).parent),
        "lock_hash": _lock_hash(settings, spec.backend),
    }
    if not run.dataset_hashes:
        values["dataset_hashes"] = {role: ref.hash for role, ref in cfg.datasets.items()}
    params = {"method": spec.method, "hparams": spec.hparams, "base_model": spec.base_model, "export": spec.export}
    tags = {
        "backend": spec.backend,
        "base_model": spec.base_model,
        "commercial_ok": str(run.commercial_ok),
        "git_sha": str(values["git_sha"]),
        "task": spec.task,
    }
    mlflow_id = _safe_forward(forwarder.start, run.id, params, tags)
    if mlflow_id is not None:
        values["mlflow_run_id"] = mlflow_id
    await _update_run(session_factory, run_id, **values)

    offset = 0
    if run.resume or run.attempt > 1:
        _, offset = read_metrics(rd, 0)  # skip metrics already forwarded by earlier attempts
    stderr_path = rd.logs_dir / "stderr.log"
    for phase in PHASES:
        if await _cancel_requested(session_factory, run_id):
            return "cancelled"
        await _update_run(session_factory, run_id, state=PHASE_STATE[phase])
        start = stderr_path.stat().st_size if stderr_path.exists() else 0
        code, offset, cancelled = await _run_phase(session_factory, run_id, settings, python, rd, phase, forwarder, offset)
        if cancelled:
            return "cancelled"
        if code != 0:
            text = _read_from(stderr_path, start)
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            error = lines[-1] if lines else f"{phase} exited with code {code}"
            await _update_run(session_factory, run_id, error=error, oom=bool(_OOM_RE.search(text)))
            return "failed"
    return "done"
