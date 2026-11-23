import asyncio
import json
import logging

import pytest

from mlops_api.models import Ingest, Upload
from mlops_worker import __main__ as wmain
from mlops_worker import executor
from mlops_worker.__main__ import cpu_tick, gpu_tick, main
from mlops_worker.executor import NullForwarder
from mlops_worker.lock import acquire, current

from .conftest import get_ingest, get_run
from .test_executor import _spec, queued_run  # noqa: F401  (seeding helper)


@pytest.fixture(autouse=True)
def fast_poll(monkeypatch):
    monkeypatch.setattr(executor, "POLL_S", 0.05)


class Env:
    pass


@pytest.fixture
def env(settings, session_factory):
    e = Env()
    e.settings, e.sf, e.fwd = settings, session_factory, NullForwarder()
    settings.poll_interval_s = 0.05
    return e


async def pending_ingest(env, name="ing-ds", n=2) -> int:
    rows = [
        {"messages": [{"role": "user", "content": f"প্রশ্ন {i}"}, {"role": "assistant", "content": f"উত্তর {i}"}]}
        for i in range(n)
    ]
    data = ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8")
    async with env.sf() as s:
        up = Upload(id=f"up-{name}", filename="a.jsonl", length=len(data), state="complete")
        s.add(up)
        d = env.settings.incoming_dir / up.id
        d.mkdir(parents=True)
        (d / "a.jsonl").write_bytes(data)
        ing = Ingest(upload_id=up.id, dataset_name=name, task="llm-sft", state="pending")
        s.add(ing)
        await s.commit()
        return ing.id


async def seed_queued_run(env, hparams):
    """Seed datasets/recipe/run and leave the run queued (queued_run claims it)."""
    run = await queued_run(env, hparams=hparams)
    async with env.sf() as s:
        from mlops_worker.lock import release

        await release(s, "run", run.id)
        r = await s.get(type(run), run.id)
        r.state = "queued"
        await s.commit()
    return run


async def test_cpu_tick_runs_pending_ingest_to_done(env):
    iid = await pending_ingest(env)
    assert await cpu_tick(env.sf, env.settings) is True
    async with env.sf() as s:
        assert (await get_ingest(s, iid)).state == "done"
    assert await cpu_tick(env.sf, env.settings) is False


async def test_gpu_tick_false_when_idle_true_after_run(env):
    assert await gpu_tick(env.sf, env.settings, lambda s: env.fwd) is False
    run = await seed_queued_run(env, {"steps": 2})
    assert await gpu_tick(env.sf, env.settings, lambda s: env.fwd) is True
    async with env.sf() as s:
        assert (await get_run(s, run.id)).state == "done"
        assert (await current(s)).holder_kind is None


async def test_ingest_runs_while_run_holds_gpu_lock(env):
    run = await seed_queued_run(env, {"steps": 1000, "step_sleep_s": 0.05})
    gpu = asyncio.create_task(gpu_tick(env.sf, env.settings, lambda s: env.fwd))
    try:
        for _ in range(200):
            async with env.sf() as s:
                if (await get_run(s, run.id)).state == "training":
                    break
            await asyncio.sleep(0.05)
        iid = await pending_ingest(env)
        assert await asyncio.wait_for(cpu_tick(env.sf, env.settings), 30) is True
        async with env.sf() as s:
            assert (await get_ingest(s, iid)).state == "done"
            assert (await current(s)).holder_id == run.id
            assert (await get_run(s, run.id)).state == "training"
    finally:
        async with env.sf() as s:
            r = await s.get(type(run), run.id)
            r.cancel_requested = True
            await s.commit()
        await asyncio.wait_for(gpu, 30)


async def test_main_processes_work_and_cancels_cleanly(env):
    iid = await pending_ingest(env)
    run = await seed_queued_run(env, {"steps": 2})
    task = asyncio.create_task(main(env.settings, forwarder_factory=lambda s: env.fwd, install_signals=False))
    try:
        async def _done():
            while True:
                async with env.sf() as s:
                    if (await get_ingest(s, iid)).state == "done" and (await get_run(s, run.id)).state == "done":
                        return
                await asyncio.sleep(0.05)

        await asyncio.wait_for(_done(), 60)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_loop_survives_tick_exception(caplog):
    calls = []

    async def tick():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        if len(calls) >= 3:
            raise asyncio.CancelledError
        return False

    with caplog.at_level(logging.ERROR):
        with pytest.raises(asyncio.CancelledError):
            await wmain.loop_forever(tick, 0.01, "test")
    assert len(calls) == 3
    assert any("boom" in (r.exc_text or "") for r in caplog.records)


async def test_main_recovers_orphaned_run(env):
    run = await seed_queued_run(env, {"steps": 1})
    async with env.sf() as s:
        r = await s.get(type(run), run.id)
        r.state = "training"
        await s.commit()
        assert await acquire(s, "run", run.id)
    task = asyncio.create_task(main(env.settings, forwarder_factory=lambda s: env.fwd, install_signals=False))
    try:
        for _ in range(100):
            async with env.sf() as s:
                r = await get_run(s, run.id)
            if r.state == "failed":
                break
            await asyncio.sleep(0.05)
        assert r.state == "failed" and "worker restarted" in r.error
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
