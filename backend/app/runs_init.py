"""Server side of warm start (spec 14): resolve and validate `init_from`, link the source checkpoint."""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .core.backend_kit import read_init_meta, resolve_checkpoint
from .core.contract import InitRef, RunDir
from .core.recipes import INIT_METHOD_PAIRS, RecipeSpec, parse_dataset_ref
from .models import ModelVersion, RecipeVersion, Run

FAKE_PAIRS = INIT_METHOD_PAIRS | {("none", "none")}  # the fake backend has method type "none"
# init_meta formats a new run cannot start from, with the reason (completes "checkpoint '<name>' ...")
UNSTARTABLE_FORMATS = {"cosyvoice-component": "is a single CosyVoice component; use 'final'"}


async def resolve_init(
    session: AsyncSession, spec: RecipeSpec, settings
) -> tuple[tuple[Run, Path, InitRef] | None, list[str]]:
    """Returns ((source run, source checkpoint dir, InitRef), []) or (None, errors); (None, []) without init_from.

    The InitRef path is the source checkpoint; the executor points it at the run's own init/ dir."""
    init = spec.init_from
    if init is None:
        return None, []
    which: str | int = init.checkpoint
    if init.run is not None:
        run = await session.get(Run, init.run)
        if run is None:
            return None, [f"init_from: run {init.run} not found"]
    else:
        run = None
        try:
            name, number = parse_dataset_ref(init.model_version)
        except ValueError:
            return None, [f"init_from: model version {init.model_version} not found"]
        mv = (await session.execute(
            select(ModelVersion).where(ModelVersion.name == name, ModelVersion.number == number))).scalar_one_or_none()
        if mv is not None:
            run = await session.get(Run, mv.run_id)
        if run is None:
            return None, [f"init_from: model version {init.model_version} not found"]
        which = "final"

    rv = await session.get(RecipeVersion, run.recipe_version_id)
    src = rv.spec if rv is not None else {}
    errors = [
        f"init_from: run {run.id} is {field} '{src.get(field)}', recipe is {field} '{getattr(spec, field)}'"
        for field in ("task", "backend", "base_model") if src.get(field) != getattr(spec, field)
    ]
    if errors:
        return None, errors

    ckpt = resolve_checkpoint(RunDir(Path(run.run_dir)), which)
    if ckpt is None:
        return None, [f"init_from: run {run.id} has no complete checkpoint '{which}'"]
    meta = read_init_meta(ckpt)
    if meta is None:
        return None, [f"init_from: run {run.id} checkpoint '{which}' has no init_meta.json"]
    reason = UNSTARTABLE_FORMATS.get(str(meta.get("format", "")))
    if reason:
        return None, [f"init_from: checkpoint '{ckpt.name}' {reason}"]
    src_method = meta.get("method") or (src.get("method") or {}).get("type")
    dst_method = spec.method["type"]
    pairs = FAKE_PAIRS if spec.backend == "fake" else INIT_METHOD_PAIRS
    if (src_method, dst_method) not in pairs:
        return None, [f"init_from: method '{src_method}' cannot start a '{dst_method}' run"]
    ref = InitRef(path=str(ckpt), source_run=run.id, checkpoint=ckpt.name, method=src_method,
                  format=str(meta.get("format", "")))
    return (run, ckpt, ref), []


def _startable(rd: RunDir, d: Path, step, meta: dict | None) -> bool:
    """`final`, or a dir that `init_from.checkpoint: <step>` resolves to (the same dir) and whose format a run
    can start from. Others (cosyvoice per-component dirs, names whose digits differ from the step) are hidden."""
    if d.name == "final":
        return True
    if meta is not None and str(meta.get("format", "")) in UNSTARTABLE_FORMATS:
        return False
    if not isinstance(step, int):
        return False
    found = resolve_checkpoint(rd, step)
    return found is not None and found.resolve() == d.resolve()


def list_checkpoints(rd: RunDir) -> list[dict]:
    """Checkpoint directories a run can start from: `final` first, then the rest newest (highest step) first.
    Only `final` and dirs that `resolve_checkpoint(rd, step)` finds again are listed (see _startable)."""
    if not rd.checkpoints_dir.is_dir():
        return []
    items = []
    for d in rd.checkpoints_dir.iterdir():
        if not d.is_dir():
            continue
        meta = read_init_meta(d)
        step = meta.get("step") if meta else None
        if not isinstance(step, int) or isinstance(step, bool):
            m = re.search(r"(\d+)$", d.name)
            step = int(m.group(1)) if m else None
        if not _startable(rd, d, step, meta):
            continue
        items.append({"name": d.name, "step": step, "complete": (d / ".complete").exists(), "init_meta": meta})
    items.sort(key=lambda i: (i["name"] != "final", -(i["step"] if i["step"] is not None else -1), i["name"]))
    return items


def link_tree(src: Path, dst: Path) -> None:
    """Hard-link every file of src into dst (copy when linking fails), built in dst.tmp then renamed."""
    src, dst = Path(src), Path(dst)
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for p in sorted(src.rglob("*")):
        target = tmp / p.relative_to(src)
        if p.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(p, target)
        except OSError:
            shutil.copy2(p, target)
    os.replace(tmp, dst)
