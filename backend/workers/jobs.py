"""Database-backed job queue: claim runs and ingests, recover after a restart."""

from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Ingest, Run, utcnow

from app.core.contract import RunDir

from .executor import kill_recorded_backend
from .lock import acquire, current, release

ACTIVE_RUN_STATES = ("preparing", "training", "evaluating", "exporting")


async def claim_next_run(session: AsyncSession) -> Run | None:
    """Move the oldest queued run to `preparing` and take the GPU lock for it."""
    run_id = (
        await session.execute(
            select(Run.id).where(Run.state == "queued").order_by(Run.created_at, Run.id).limit(1)
        )
    ).scalar_one_or_none()
    if run_id is None:
        return None
    if not await acquire(session, "run", run_id):
        return None
    result = await session.execute(
        update(Run)
        .where(Run.id == run_id, Run.state == "queued")
        .values(state="preparing", started_at=utcnow())
    )
    await session.commit()
    if result.rowcount != 1:
        await release(session, "run", run_id)
        return None
    return (await session.execute(select(Run).where(Run.id == run_id).execution_options(populate_existing=True))).scalar_one()


async def _pending_ingest_ids(session: AsyncSession) -> list[int]:
    rows = await session.execute(
        select(Ingest.id).where(Ingest.state == "pending").order_by(Ingest.created_at, Ingest.id)
    )
    return list(rows.scalars())


async def claim_next_ingest(session: AsyncSession) -> Ingest | None:
    """Move the oldest pending ingest to `running` (no GPU lock needed)."""
    for ingest_id in await _pending_ingest_ids(session):
        result = await session.execute(
            update(Ingest).where(Ingest.id == ingest_id, Ingest.state == "pending").values(state="running")
        )
        await session.commit()
        if result.rowcount == 1:
            return (await session.execute(select(Ingest).where(Ingest.id == ingest_id).execution_options(populate_existing=True))).scalar_one()
    return None


async def recover(session: AsyncSession, cancel_grace_s: float = 60) -> list[str]:
    """Fail work orphaned by a worker restart and free a run-held GPU lock.

    A backend process group still alive from the dead worker (recorded in logs/backend.pid)
    is stopped first: SIGTERM, up to `cancel_grace_s`, then SIGKILL."""
    rows = (
        await session.execute(
            select(Run.id, Run.state, Run.run_dir).where(Run.state.in_(ACTIVE_RUN_STATES))
        )
    ).all()
    recovered: list[str] = []
    for run_id, state, run_dir in rows:
        await kill_recorded_backend(RunDir(Path(run_dir)), cancel_grace_s)
        result = await session.execute(
            update(Run)
            .where(Run.id == run_id, Run.state == state)
            .values(state="failed", error=f"worker restarted during {state}", finished_at=utcnow())
        )
        if result.rowcount == 1:
            recovered.append(run_id)
    await session.execute(
        update(Ingest)
        .where(Ingest.state == "running")
        .values(state="failed", error="worker restarted during ingest")
    )
    await session.commit()
    lock = await current(session)
    if lock.holder_kind == "run":
        await release(session, "run", lock.holder_id)
    return recovered
