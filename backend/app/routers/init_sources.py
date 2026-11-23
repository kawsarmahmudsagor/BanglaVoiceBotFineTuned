"""Warm start sources (spec 14.9): runs and model versions a new run may start from."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.backend_kit import resolve_checkpoint
from app.core.contract import RunDir

from ..db import get_session
from ..models import ModelVersion, Recipe, RecipeVersion, Run
from ..runs_init import list_checkpoints
from .datasets import iso

router = APIRouter(prefix="/api")
MAX_RUNS = 50


@router.get("/init-sources")
async def init_sources(task: str = Query(...), backend: str = Query(...), base_model: str = Query(...),
                       session: AsyncSession = Depends(get_session)) -> dict:
    q = (select(Run, Recipe.name, RecipeVersion.spec)
         .join(RecipeVersion, RecipeVersion.id == Run.recipe_version_id)
         .join(Recipe, Recipe.id == RecipeVersion.recipe_id)
         .order_by(Run.created_at.desc(), Run.id.desc()))
    runs, matching = [], {}
    for run, recipe, spec in (await session.execute(q)).all():
        if (spec.get("task"), spec.get("backend"), spec.get("base_model")) != (task, backend, base_model):
            continue
        matching[run.id] = run
        if len(runs) >= MAX_RUNS:
            continue
        done = [c for c in list_checkpoints(RunDir(Path(run.run_dir))) if c["complete"]]
        if done:
            runs.append({"id": run.id, "recipe": recipe, "state": run.state,
                         "finished_at": iso(run.finished_at), "checkpoints": done})
    mvs = (await session.execute(select(ModelVersion).where(ModelVersion.task == task, ModelVersion.base_model == base_model)
                                 .order_by(ModelVersion.created_at.desc(), ModelVersion.id.desc()))).scalars().all()
    return {
        "runs": runs,
        "model_versions": [{"ref": f"{m.name}@v{m.number}", "run_id": m.run_id, "metrics": m.metrics}
                           for m in mvs if m.run_id in matching and resolve_checkpoint(RunDir(Path(matching[m.run_id].run_dir)), "final")],
    }
