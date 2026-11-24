from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.contract import RunDir, read_eval_results, read_export_manifest

from ..db import get_session
from ..models import ModelVersion, Run
from .datasets import iso
from .runs import artifact_response, get_run, run_recipe

router = APIRouter(prefix="/api")


def _out(m: ModelVersion) -> dict:
    return {
        "id": m.id, "name": m.name, "number": m.number, "run_id": m.run_id, "task": m.task,
        "base_model": m.base_model, "metrics": m.metrics, "exports": m.exports,
        "commercial_ok": m.commercial_ok, "champion": m.champion, "created_at": iso(m.created_at),
    }


async def _model(session: AsyncSession, model_id: int) -> ModelVersion:
    m = await session.get(ModelVersion, model_id)
    if m is None:
        raise HTTPException(404, f"model {model_id} not found")
    return m


async def _for_run(session: AsyncSession, run_id: str) -> ModelVersion | None:
    return (await session.execute(select(ModelVersion).where(ModelVersion.run_id == run_id))).scalar_one_or_none()


async def _next_number(session: AsyncSession, name: str) -> int:
    top = (await session.execute(select(func.max(ModelVersion.number)).where(ModelVersion.name == name))
           ).scalar_one()
    return (top or 0) + 1


@router.post("/runs/{run_id}/promote", status_code=201, response_model=None)
async def promote(run_id: str, session: AsyncSession = Depends(get_session)):
    run = await get_run(session, run_id)
    if run.state != "done":
        raise HTTPException(409, f"only a done run can be promoted (state is {run.state})")
    existing = await _for_run(session, run_id)
    if existing is not None:
        return JSONResponse(_out(existing), status_code=200)
    recipe, rv = await run_recipe(session, run)
    rd = RunDir(Path(run.run_dir))
    results, manifest = read_eval_results(rd), read_export_manifest(rd)
    # copy plain values: a rollback expires ORM instances
    name, task, base = recipe.name, rv.spec["task"], rv.spec["base_model"]
    commercial = run.commercial_ok
    for attempt in (1, 2):
        m = ModelVersion(
            name=name, number=await _next_number(session, name), run_id=run_id,
            task=task, base_model=base,
            metrics=results.metrics if results else {},
            exports=[a.model_dump(mode="json") for a in manifest.artifacts] if manifest else [],
            commercial_ok=commercial,
        )
        session.add(m)
        try:
            await session.commit()
            return _out(m)
        except IntegrityError:
            await session.rollback()
            existing = await _for_run(session, run_id)  # concurrent promote of this same run
            if existing is not None:
                return JSONResponse(_out(existing), status_code=200)
            if attempt == 2:  # another run of this recipe took the number twice
                raise


@router.get("/models")
async def list_models(task: str | None = None, session: AsyncSession = Depends(get_session)) -> list[dict]:
    q = select(ModelVersion).order_by(ModelVersion.created_at.desc(), ModelVersion.id.desc())
    if task:
        q = q.where(ModelVersion.task == task)
    return [_out(m) for m in (await session.execute(q)).scalars()]


@router.get("/models/{model_id}")
async def get_model(model_id: int, session: AsyncSession = Depends(get_session)) -> dict:
    return _out(await _model(session, model_id))


@router.post("/models/{model_id}/champion")
async def set_champion(model_id: int, session: AsyncSession = Depends(get_session)) -> dict:
    m = await _model(session, model_id)
    await session.execute(
        update(ModelVersion).where(ModelVersion.task == m.task, ModelVersion.id != m.id).values(champion=False))
    m.champion = True
    await session.commit()
    return _out(m)


@router.get("/models/{model_id}/exports/{index}")
async def download_export(model_id: int, index: int, session: AsyncSession = Depends(get_session)) -> Response:
    m = await _model(session, model_id)
    if not 0 <= index < len(m.exports):
        raise HTTPException(404, "export not found")
    run = await session.get(Run, m.run_id)
    export_dir = RunDir(Path(run.run_dir)).export_dir
    target = export_dir / m.exports[index]["path"]
    try:
        return artifact_response(target, (export_dir,), export_dir)
    except HTTPException:
        raise HTTPException(404, "export file not found") from None
