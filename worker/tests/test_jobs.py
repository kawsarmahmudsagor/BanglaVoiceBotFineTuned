import datetime as dt
import signal
import subprocess
import sys

import pytest

from mlops_worker import jobs
from mlops_worker.jobs import claim_next_ingest, claim_next_run, recover
from mlops_worker.lock import acquire, current, release

from .conftest import get_ingest, get_run, make_ingest, make_run

T = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


async def test_claim_takes_oldest_and_holds_lock(session):
    newer = await make_run(session, created_at=T + dt.timedelta(hours=2))
    older = await make_run(session, created_at=T + dt.timedelta(hours=1))
    claimed = await claim_next_run(session)
    assert claimed.id == older.id
    assert claimed.state == "preparing"
    assert claimed.started_at is not None
    lock = await current(session)
    assert (lock.holder_kind, lock.holder_id) == ("run", older.id)
    assert (await get_run(session, newer.id)).state == "queued"
    # lock held -> nothing more is claimed
    assert await claim_next_run(session) is None


async def test_claim_ties_broken_by_id(session):
    await make_run(session, created_at=T, run_id="r-0000000b")
    a = await make_run(session, created_at=T, run_id="r-0000000a")
    assert (await claim_next_run(session)).id == a.id


async def test_claim_returns_none_while_serve_holds_lock(session):
    run = await make_run(session)
    await acquire(session, "serve", "llm-1")
    assert await claim_next_run(session) is None
    assert (await get_run(session, run.id)).state == "queued"
    assert (await current(session)).holder_id == "llm-1"


async def test_claim_returns_none_when_no_queued_and_leaves_lock_free(session):
    await make_run(session, state="done")
    assert await claim_next_run(session) is None
    assert (await current(session)).holder_kind is None


async def test_claim_lost_race_releases_lock(session, session_factory, monkeypatch):
    run = await make_run(session)
    real_acquire = jobs.acquire

    async def acquire_then_steal(s, kind, holder_id):
        ok = await real_acquire(s, kind, holder_id)
        async with session_factory() as other:  # another worker claims it first
            r = await get_run(other, run.id)
            r.state = "preparing"
            await other.commit()
        return ok

    monkeypatch.setattr(jobs, "acquire", acquire_then_steal)
    assert await claim_next_run(session) is None
    assert (await current(session)).holder_kind is None


async def test_claim_next_ingest_oldest_pending(session):
    await make_ingest(session, state="done")
    newer = await make_ingest(session, created_at=T + dt.timedelta(hours=2))
    older = await make_ingest(session, created_at=T + dt.timedelta(hours=1))
    got = await claim_next_ingest(session)
    assert got.id == older.id and got.state == "running"
    assert (await get_ingest(session, newer.id)).state == "pending"
    assert (await current(session)).holder_kind is None  # no GPU lock
    assert (await claim_next_ingest(session)).id == newer.id
    assert await claim_next_ingest(session) is None


async def test_claim_next_ingest_retries_after_losing_race(session, session_factory, monkeypatch):
    first = await make_ingest(session, created_at=T)
    second = await make_ingest(session, created_at=T + dt.timedelta(hours=1))
    real = jobs._pending_ingest_ids
    calls = 0

    async def stale_candidates(s):
        nonlocal calls
        ids = await real(s)
        calls += 1
        if calls == 1:
            async with session_factory() as other:
                i = await get_ingest(other, first.id)
                i.state = "running"
                await other.commit()
        return ids

    monkeypatch.setattr(jobs, "_pending_ingest_ids", stale_candidates)
    got = await claim_next_ingest(session)
    assert got.id == second.id and got.state == "running"


async def test_recover_marks_running_failed_and_frees_lock(session):
    run = await make_run(session, state="training")
    await acquire(session, "run", run.id)
    assert await recover(session) == [run.id]
    r = await get_run(session, run.id)
    assert r.state == "failed"
    assert r.error == "worker restarted during training"
    assert r.finished_at is not None
    assert (await current(session)).holder_kind is None


async def test_recover_leaves_other_states_and_serve_lock(session):
    queued = await make_run(session, state="queued")
    done = await make_run(session, state="done")
    stuck = await make_run(session, state="exporting")
    running_ingest = await make_ingest(session, state="running")
    pending_ingest = await make_ingest(session, state="pending")
    await acquire(session, "serve", "llm-1")
    assert await recover(session) == [stuck.id]
    assert (await get_run(session, queued.id)).state == "queued"
    assert (await get_run(session, done.id)).state == "done"
    ri = await get_ingest(session, running_ingest.id)
    assert ri.state == "failed" and ri.error == "worker restarted during ingest"
    assert (await get_ingest(session, pending_ingest.id)).state == "pending"
    assert (await current(session)).holder_id == "llm-1"
    await release(session, "serve", "llm-1")


async def test_recover_covers_all_active_states(session):
    ids = [(await make_run(session, state=s)).id for s in ("preparing", "training", "evaluating", "exporting")]
    assert sorted(await recover(session)) == sorted(ids)


async def test_recover_without_pid_file_still_fails_run(session, tmp_path):
    run = await make_run(session, state="training")
    run.run_dir = str(tmp_path / "run")
    await session.commit()
    assert await recover(session, cancel_grace_s=1) == [run.id]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
async def test_recover_kills_orphaned_backend_group(session, tmp_path):
    run = await make_run(session, state="training")
    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    run.run_dir = str(run_dir)
    await session.commit()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        (run_dir / "logs" / "backend.pid").write_text(str(child.pid))
        assert await recover(session, cancel_grace_s=1) == [run.id]
        assert child.wait(timeout=5) == -signal.SIGTERM
        assert not (run_dir / "logs" / "backend.pid").exists()
        assert (await get_run(session, run.id)).state == "failed"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
