import asyncio
import json
import sys

import pytest
from sqlalchemy import select

from app.models import Dataset, DatasetVersion, Recipe, RecipeVersion, Run
from app.core.contract import Metric, RunDir, read_config
from workers import executor
from workers.executor import PHASE_STATE, NullForwarder, build_config, execute_run
from workers.jobs import claim_next_run
from workers.lock import current

from .conftest import get_run


class RecordingForwarder:
    def __init__(self):
        self.started = []
        self.metrics_seen: list[Metric] = []
        self.finished: list[str] = []
        self.params_seen: list[dict] = []

    def params(self, params):
        self.params_seen.append(params)

    def start(self, run_id, params, tags):
        self.started.append((run_id, params, tags))
        return "mlflow-1"

    def metrics(self, items):
        self.metrics_seen.extend(items)

    def finish(self, state):
        self.finished.append(state)


class Env:
    pass


def _spec(hparams, eval_ref=None, backend="fake"):
    return {
        "name": "rec", "task": "llm-sft", "backend": backend, "base_model": "fake/base",
        "base_model_meta": {"license": "apache-2.0", "commercial_ok": True},
        "datasets": {"train": "trainset@v1", **({"eval": eval_ref} if eval_ref else {})},
        "method": {"type": "full"}, "hparams": hparams, "export": ["dummy"],
    }


@pytest.fixture(autouse=True)
def fast_poll(monkeypatch):
    monkeypatch.setattr(executor, "POLL_S", 0.05)


@pytest.fixture
def env(settings, session_factory):
    e = Env()
    e.settings, e.sf, e.fwd = settings, session_factory, RecordingForwarder()
    return e


async def queued_run(env, hparams=None, eval_only_eval=False, backend="fake"):
    """Create datasets + recipe + a run, then claim it (state preparing, lock held)."""
    async with env.sf() as s:
        ds = Dataset(name="trainset", task="llm-sft")
        s.add(ds)
        await s.flush()
        s.add(DatasetVersion(dataset_id=ds.id, number=1, hash="h-train", path=str(env.settings.data_root / "ds-train"), n_samples=1))
        eval_ref = None
        if eval_only_eval:
            ev = Dataset(name="evalset", task="llm-sft", eval_only=True)
            s.add(ev)
            await s.flush()
            s.add(DatasetVersion(dataset_id=ev.id, number=1, hash="h-eval", path=str(env.settings.data_root / "ds-eval"), n_samples=1))
            eval_ref = "evalset@v1"
        recipe = Recipe(name="rec")
        s.add(recipe)
        await s.flush()
        rv = RecipeVersion(recipe_id=recipe.id, number=1, spec=_spec(hparams or {}, eval_ref, backend), yaml="x")
        s.add(rv)
        await s.flush()
        run_dir = env.settings.runs_dir / "r-test0001"
        s.add(Run(id="r-test0001", recipe_version_id=rv.id, run_dir=str(run_dir), commercial_ok=True))
        await s.commit()
        run = await claim_next_run(s)
        assert run is not None
        return run


async def lock_state(env):
    async with env.sf() as s:
        return await current(s)


async def set_cancel(env, run_id):
    async with env.sf() as s:
        run = await s.get(Run, run_id)
        run.cancel_requested = True
        await s.commit()


async def wait_for_state(env, run_id, state, timeout=10):
    async def _w():
        while True:
            async with env.sf() as s:
                if (await get_run(s, run_id)).state == state:
                    return
            await asyncio.sleep(0.02)
    await asyncio.wait_for(_w(), timeout)


async def run_of(env, run_id):
    async with env.sf() as s:
        return await get_run(s, run_id)


async def test_successful_run_reaches_done(env):
    run = await queued_run(env, hparams={"steps": 3})
    assert await execute_run(env.sf, run.id, env.settings, env.fwd) == "done"
    assert [m.step for m in env.fwd.metrics_seen] == [1, 2, 3]
    assert (await lock_state(env)).holder_kind is None
    r = await run_of(env, run.id)
    assert r.state == "done" and r.finished_at is not None
    assert r.dataset_hashes == {"train": "h-train"}
    assert r.mlflow_run_id == "mlflow-1"
    assert env.fwd.finished == ["done"]
    cfg = read_config(RunDir(r.run_dir))
    assert cfg.datasets["train"].manifest.endswith("manifest.parquet")
    _, params, tags = env.fwd.started[0]
    assert tags["backend"] == "fake" and tags["commercial_ok"] == "True" and tags["task"] == "llm-sft"
    assert params["base_model"] == "fake/base"
    assert (RunDir(r.run_dir).logs_dir / "stdout.log").exists()


async def test_failed_phase_records_error_and_oom(env):
    run = await queued_run(env, hparams={"oom": True})
    assert await execute_run(env.sf, run.id, env.settings, env.fwd) == "failed"
    r = await run_of(env, run.id)
    assert r.oom and "CUDA out of memory" in r.error
    assert (await lock_state(env)).holder_kind is None
    assert env.fwd.finished == ["failed"]


async def test_failed_phase_without_oom(env):
    run = await queued_run(env, hparams={"fail_phase": "eval"})
    assert await execute_run(env.sf, run.id, env.settings, env.fwd) == "failed"
    r = await run_of(env, run.id)
    assert not r.oom and "forced failure in eval" in r.error


async def test_cancel_before_phase_start(env):
    run = await queued_run(env)
    await set_cancel(env, run.id)
    assert await execute_run(env.sf, run.id, env.settings, env.fwd) == "cancelled"
    assert (await lock_state(env)).holder_kind is None


async def test_cancel_terminates_running_backend(env):
    env.settings.cancel_grace_s = 5
    run = await queued_run(env, hparams={"steps": 1000, "step_sleep_s": 0.05})
    task = asyncio.create_task(execute_run(env.sf, run.id, env.settings, env.fwd))
    await wait_for_state(env, run.id, "training")
    await set_cancel(env, run.id)
    assert await asyncio.wait_for(task, 15) == "cancelled"
    assert (await lock_state(env)).holder_kind is None
    assert (await run_of(env, run.id)).state == "cancelled"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
async def test_cancel_sigkills_backend_that_ignores_sigterm(env):
    env.settings.cancel_grace_s = 1
    run = await queued_run(env, hparams={"steps": 1000, "step_sleep_s": 0.05, "ignore_sigterm": True})
    task = asyncio.create_task(execute_run(env.sf, run.id, env.settings, env.fwd))
    await wait_for_state(env, run.id, "training")
    await set_cancel(env, run.id)
    assert await asyncio.wait_for(task, 10) == "cancelled"
    assert (await lock_state(env)).holder_kind is None


async def test_build_config_eval_splits(env):
    from app.core.recipes import RecipeSpec

    run = await queued_run(env, eval_only_eval=True)
    async with env.sf() as s:
        versions = {}
        for name in ("trainset", "evalset"):
            ds = (await s.execute(select(Dataset).where(Dataset.name == name))).scalar_one()
            versions[f"{name}@v1"] = (await s.execute(select(DatasetVersion).where(DatasetVersion.dataset_id == ds.id))).scalar_one()
    spec = RecipeSpec.model_validate(_spec({"seed": 7}, "evalset@v1"))
    cfg = build_config(run, spec, versions, env.settings, eval_only={"evalset"})
    assert cfg.datasets["train"].splits == ["train", "val"]
    assert cfg.datasets["eval"].splits == ["train", "val", "test"]
    assert cfg.seed == 7 and cfg.prepared_root == str(env.settings.prepared_dir)

    # eval dataset that is not eval_only -> only the test split
    spec2 = RecipeSpec.model_validate(_spec({}, "trainset@v1"))
    cfg2 = build_config(run, spec2, {"trainset@v1": versions["trainset@v1"]}, env.settings, eval_only=set())
    assert cfg2.datasets["eval"].splits == ["test"]


async def test_missing_venv_fails_run_with_message(env, monkeypatch):
    monkeypatch.delenv("BACKEND_PYTHON_NOBACKEND", raising=False)
    run = await queued_run(env, backend="nobackend")
    assert await execute_run(env.sf, run.id, env.settings, env.fwd) == "failed"
    r = await run_of(env, run.id)
    assert "setup_venvs.sh" in r.error
    assert (await lock_state(env)).holder_kind is None


async def test_forwarder_exceptions_do_not_fail_run(env):
    class Broken(NullForwarder):
        def start(self, *a):
            raise RuntimeError("mlflow down")

        def metrics(self, items):
            raise RuntimeError("mlflow down")

    run = await queued_run(env, hparams={"steps": 2})
    assert await execute_run(env.sf, run.id, env.settings, Broken()) == "done"


def test_phase_state_mapping():
    assert PHASE_STATE == {"prepare": "preparing", "train": "training", "eval": "evaluating", "export": "exporting"}


async def test_resume_does_not_replay_earlier_metrics(env):
    from workers.lock import acquire

    run = await queued_run(env, hparams={"steps": 3})
    assert await execute_run(env.sf, run.id, env.settings, env.fwd) == "done"
    assert len(env.fwd.metrics_seen) == 3
    async with env.sf() as s:
        r = await s.get(Run, run.id)
        r.state, r.resume, r.attempt, r.finished_at = "preparing", True, 2, None
        await s.commit()
        assert await acquire(s, "run", run.id)
    fwd2 = RecordingForwarder()
    assert await execute_run(env.sf, run.id, env.settings, fwd2) == "done"
    assert fwd2.metrics_seen == []


async def test_task_cancellation_records_interruption_and_releases_lock(env):
    env.settings.cancel_grace_s = 5
    run = await queued_run(env, hparams={"steps": 1000, "step_sleep_s": 0.05})
    task = asyncio.create_task(execute_run(env.sf, run.id, env.settings, env.fwd))
    await wait_for_state(env, run.id, "training")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    r = await run_of(env, run.id)
    assert r.state == "failed" and r.error == "worker interrupted"
    assert (await lock_state(env)).holder_kind is None


async def test_exception_error_text_has_type_name(env, monkeypatch):
    monkeypatch.delenv("BACKEND_PYTHON_NOBACKEND", raising=False)
    run = await queued_run(env, backend="nobackend")
    await execute_run(env.sf, run.id, env.settings, env.fwd)
    assert (await run_of(env, run.id)).error.startswith("BackendMissing: ")


async def test_blocking_forwarder_does_not_block_event_loop(env):
    import time

    class SlowForwarder(RecordingForwarder):
        def start(self, run_id, params, tags):
            time.sleep(0.5)
            return super().start(run_id, params, tags)

        def metrics(self, items):
            time.sleep(0.5)
            super().metrics(items)

        def finish(self, state):
            time.sleep(0.5)
            super().finish(state)

    max_gap = 0.0
    stop = False

    async def ticker():
        nonlocal max_gap
        last = time.monotonic()
        while not stop:
            await asyncio.sleep(0.01)
            now = time.monotonic()
            max_gap = max(max_gap, now - last)
            last = now

    run = await queued_run(env, hparams={"steps": 3})
    t = asyncio.create_task(ticker())
    assert await execute_run(env.sf, run.id, env.settings, SlowForwarder()) == "done"
    stop = True
    await t
    # each blocking forwarder call sleeps 0.5 s; the loop must keep ticking meanwhile
    assert max_gap < 0.3, max_gap


async def test_backend_pid_file_written_during_phase_and_removed_after(env):
    run = await queued_run(env, hparams={"steps": 1000, "step_sleep_s": 0.05})
    pid_file = RunDir(env.settings.runs_dir / "r-test0001").logs_dir / "backend.pid"
    task = asyncio.create_task(execute_run(env.sf, run.id, env.settings, env.fwd))
    await wait_for_state(env, run.id, "training")
    for _ in range(200):
        if pid_file.exists() and pid_file.read_text().strip():
            break
        await asyncio.sleep(0.02)
    recorded = json.loads(pid_file.read_text())
    assert recorded["pgid"] > 0
    if sys.platform == "linux":
        assert recorded["boot_id"] and isinstance(recorded["start_ticks"], int)
    await set_cancel(env, run.id)
    assert await asyncio.wait_for(task, 15) == "cancelled"
    assert not pid_file.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
async def test_worker_cancellation_sigterms_backend_before_kill(env):
    env.settings.cancel_grace_s = 10
    run = await queued_run(env, hparams={"steps": 1000, "step_sleep_s": 30})
    rd = RunDir(env.settings.runs_dir / "r-test0001")
    task = asyncio.create_task(execute_run(env.sf, run.id, env.settings, env.fwd))
    await wait_for_state(env, run.id, "training")
    for _ in range(500):  # the fake installs its SIGTERM handler right after creating checkpoints/
        if rd.checkpoints_dir.exists():
            break
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert list(rd.checkpoints_dir.glob("step-*")), "backend was killed without SIGTERM"
    r = await run_of(env, run.id)
    assert r.state == "failed" and r.error == "worker interrupted"
    assert (await lock_state(env)).holder_kind is None


async def test_prepared_refs_forwarded_after_prepare(env):
    run = await queued_run(env, hparams={"steps": 1, "write_prepared_refs": True})
    assert await execute_run(env.sf, run.id, env.settings, env.fwd) == "done"
    assert env.fwd.params_seen == [{
        "prepared.train.spec_hash": "spec-train",
        "prepared.train.cache_hit": "True",
        "prepared.eval.spec_hash": "spec-eval",
        "prepared.eval.cache_hit": "False",
    }]


async def test_no_params_forwarded_without_prepared_refs(env):
    run = await queued_run(env, hparams={"steps": 1})
    assert await execute_run(env.sf, run.id, env.settings, env.fwd) == "done"
    assert env.fwd.params_seen == []


async def test_hf_home_passed_to_backend(env, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    run = await queued_run(env, hparams={"steps": 1, "echo_env": True})
    assert await execute_run(env.sf, run.id, env.settings, env.fwd) == "done"
    out = (RunDir(run.run_dir).logs_dir / "stdout.log").read_text()
    assert str(env.settings.hf_home) in out
    assert out.replace("\\", "/").count("models/hf") >= 1


async def test_existing_hf_home_is_respected(env, monkeypatch):
    monkeypatch.setenv("HF_HOME", "/custom/hf")
    run = await queued_run(env, hparams={"steps": 1, "echo_env": True})
    assert await execute_run(env.sf, run.id, env.settings, env.fwd) == "done"
    assert "HF_HOME=/custom/hf" in (RunDir(run.run_dir).logs_dir / "stdout.log").read_text()
