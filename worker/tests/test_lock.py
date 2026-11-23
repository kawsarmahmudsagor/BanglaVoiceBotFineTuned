from sqlalchemy import delete

from mlops_api.models import GpuLock
from mlops_worker.lock import acquire, current, ensure_lock_row, release


async def test_lock_is_exclusive(session):
    assert await acquire(session, "run", "r-1")
    assert not await acquire(session, "serve", "llm-1")
    await release(session, "serve", "llm-1")  # not holder: no-op
    assert (await current(session)).holder_id == "r-1"
    await release(session, "run", "r-1")
    assert await acquire(session, "serve", "llm-1")


async def test_acquire_sets_acquired_at_and_release_clears_all(session):
    assert await acquire(session, "run", "r-1")
    lock = await current(session)
    assert lock.acquired_at is not None
    await release(session, "run", "r-1")
    lock = await current(session)
    assert (lock.holder_kind, lock.holder_id, lock.acquired_at) == (None, None, None)


async def test_release_requires_kind_and_id_match(session):
    await acquire(session, "run", "r-1")
    await release(session, "serve", "r-1")
    await release(session, "run", "r-2")
    assert (await current(session)).holder_id == "r-1"


async def test_ensure_lock_row_idempotent(session):
    await session.execute(delete(GpuLock))
    await session.commit()
    await ensure_lock_row(session)
    await ensure_lock_row(session)
    assert (await current(session)).id == 1
    await acquire(session, "run", "r-1")
    await ensure_lock_row(session)  # must not clobber holder
    assert (await current(session)).holder_id == "r-1"
