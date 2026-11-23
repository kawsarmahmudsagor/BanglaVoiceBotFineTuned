"""Single-holder GPU lock backed by the `gpu_lock` row (id=1)."""

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import GpuLock, utcnow

LOCK_ID = 1


async def ensure_lock_row(session: AsyncSession) -> None:
    """Create the lock row if absent (idempotent; never touches an existing holder)."""
    if await session.get(GpuLock, LOCK_ID) is None:
        session.add(GpuLock(id=LOCK_ID))
        await session.commit()


async def acquire(session: AsyncSession, kind: str, holder_id: str) -> bool:
    result = await session.execute(
        update(GpuLock)
        .where(GpuLock.id == LOCK_ID, GpuLock.holder_kind.is_(None))
        .values(holder_kind=kind, holder_id=holder_id, acquired_at=utcnow())
    )
    await session.commit()
    return result.rowcount == 1


async def release(session: AsyncSession, kind: str, holder_id: str) -> None:
    await session.execute(
        update(GpuLock)
        .where(GpuLock.id == LOCK_ID, GpuLock.holder_kind == kind, GpuLock.holder_id == holder_id)
        .values(holder_kind=None, holder_id=None, acquired_at=None)
    )
    await session.commit()


async def current(session: AsyncSession) -> GpuLock:
    return (await session.execute(select(GpuLock).where(GpuLock.id == LOCK_ID).execution_options(populate_existing=True))).scalar_one()
