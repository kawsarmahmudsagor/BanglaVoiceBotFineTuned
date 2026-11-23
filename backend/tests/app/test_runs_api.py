import asyncio
import json
import os
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app.models import Dataset, Recipe, RecipeVersion, Run
from app.routers import runs as runs_router
from app.core.contract import (
    Metric, RunDir, Status, append_eval_sample, append_metric, write_prepared_refs, write_status,
)
from runs_helpers import finish_files, make_run, set_state


async def test_create_run_freezes_recipe_and_sets_commercial_flag(client, session, settings):
    rid = await make_run(client, session)
    assert rid.startswith("r-") and len(rid) == 10
    d = (await client.get(f"/api/runs/{rid}")).json()
    assert d["state"] == "queued" and d["commercial_ok"] is True
    assert d["recipe"] == {"name": "r1", "number": 1}
    assert d["run_dir"] == str(settings.runs_dir / rid)
    assert d["status"] is None and d["eval"] is None and d["exports"] == [] and d["oom_hint"] is None
    assert (await client.get("/api/recipes/r1/versions/1")).json()["frozen"] is True
    run = await session.get(Run, rid)
    assert run.dataset_hashes == {"train": "h1"}


async def test_create_run_commercial_false_when_dataset_or_base_not_ok(client, session):
    rid = await make_run(client, session, ds_commercial=False)
    assert (await client.get(f"/api/runs/{rid}")).json()["commercial_ok"] is False
    rid2 = await make_run(client, session, name="r2", commercial=False)
    assert (await client.get(f"/api/runs/{rid2}")).json()["commercial_ok"] is False


async def test_create_run_errors(client, session, settings):
    assert (await client.post("/api/runs", json={"recipe": "nope"})).status_code == 404
    await make_run(client, session)
    assert (await client.post("/api/runs", json={"recipe": "r1", "version": 9})).status_code == 404
    settings.disk_min_gb = 10**9
    assert (await client.post("/api/runs", json={"recipe": "r1"})).status_code == 507
    settings.disk_min_gb = 0
    ds = (await session.execute(select(Dataset))).scalar_one()
    await session.delete(ds)
    await session.commit()
    r = await client.post("/api/runs", json={"recipe": "r1"})
    assert r.status_code == 422 and "dataset 'd1@v1' not found" in r.json()["errors"]


async def test_list_runs_newest_first_and_filter(client, session, session_factory):
    a = await make_run(client, session)
    b = await make_run(client, session)
    await set_state(session_factory, a, state="failed")
    items = (await client.get("/api/runs")).json()
    assert [i["id"] for i in items] == [b, a]
    assert items[0]["recipe"] == "r1" and "commercial_ok" in items[0] and "created_at" in items[0]
    only = (await client.get("/api/runs?state=failed")).json()
    assert [i["id"] for i in only] == [a]


async def test_list_runs_reports_the_runs_own_recipe_version_and_backend(client, session):
    rid = await make_run(client, session)
    recipe = (await session.execute(select(Recipe).where(Recipe.name == "r1"))).scalar_one()
    v1 = (await session.execute(select(RecipeVersion).where(RecipeVersion.recipe_id == recipe.id))).scalar_one()
    session.add(RecipeVersion(recipe_id=recipe.id, number=2, yaml="", spec={**v1.spec, "backend": "other"}))
    await session.commit()
    (item,) = [i for i in (await client.get("/api/runs")).json() if i["id"] == rid]
    assert item["recipe_number"] == 1
    assert item["backend"] == "fake"
    assert item["task"] == "llm-sft"


async def test_detail_404_and_oom_hint(client, session, session_factory):
    assert (await client.get("/api/runs/r-nope")).status_code == 404
    rid = await make_run(client, session)
    await set_state(session_factory, rid, state="failed", oom=True)
    d = (await client.get(f"/api/runs/{rid}")).json()
    assert d["oom_hint"] == "CUDA out of memory: lower batch_size or enable gradient checkpointing"


async def test_detail_reads_run_dir_files(client, session):
    rid = await make_run(client, session)
    rd = finish_files((await client.get(f"/api/runs/{rid}")).json()["run_dir"])
    write_status(rd, Status(phase="train", step=3, updated_at=1.0))
    d = (await client.get(f"/api/runs/{rid}")).json()
    assert d["status"]["step"] == 3
    assert d["eval"]["metrics"] == {"wer": 0.2}
    assert d["exports"][0]["path"] == "model.bin"


async def test_cancel_queued_is_immediate_and_terminal_is_409(client, session):
    rid = await make_run(client, session)
    r = await client.post(f"/api/runs/{rid}/cancel")
    assert r.status_code == 200
    d = (await client.get(f"/api/runs/{rid}")).json()
    assert d["state"] == "cancelled" and d["finished_at"] is not None
    assert (await client.post(f"/api/runs/{rid}/cancel")).status_code == 409
    assert (await client.post("/api/runs/r-nope/cancel")).status_code == 404


async def test_cancel_running_requests_cancel(client, session, session_factory):
    rid = await make_run(client, session)
    await set_state(session_factory, rid, state="training")
    r = await client.post(f"/api/runs/{rid}/cancel")
    assert r.status_code == 202
    d = (await client.get(f"/api/runs/{rid}")).json()
    assert d["state"] == "training" and d["cancel_requested"] is True


async def test_resume_only_from_failed_or_cancelled(client, session, session_factory, settings):
    rid = await make_run(client, session)
    assert (await client.post(f"/api/runs/{rid}/resume")).status_code == 409
    await set_state(session_factory, rid, state="failed", error="boom", oom=True, cancel_requested=True)
    r = await client.post(f"/api/runs/{rid}/resume")
    assert r.status_code == 200
    d = (await client.get(f"/api/runs/{rid}")).json()
    assert (d["state"], d["resume"], d["attempt"], d["cancel_requested"], d["error"], d["oom"]) == (
        "queued", True, 2, False, None, False)
    await set_state(session_factory, rid, state="cancelled")
    settings.disk_min_gb = 10**9
    assert (await client.post(f"/api/runs/{rid}/resume")).status_code == 507
    settings.disk_min_gb = 0
    assert (await client.post(f"/api/runs/{rid}/resume")).status_code == 200
    assert (await client.post("/api/runs/r-nope/resume")).status_code == 404


async def test_resume_clears_finished_at(client, session, session_factory):
    import datetime as dt
    rid = await make_run(client, session)
    await set_state(session_factory, rid, state="failed", finished_at=dt.datetime.now(dt.timezone.utc))
    assert (await client.post(f"/api/runs/{rid}/resume")).status_code == 200
    assert (await client.get(f"/api/runs/{rid}")).json()["finished_at"] is None


async def test_cancel_loses_race_to_worker_claim(client, session, session_factory, monkeypatch):
    rid = await make_run(client, session)
    orig = runs_router.get_run

    async def stale_get(sess, run_id):
        run = await orig(sess, run_id)
        await set_state(session_factory, run_id, state="preparing")  # worker claim lands after our read
        return run

    monkeypatch.setattr(runs_router, "get_run", stale_get)
    r = await client.post(f"/api/runs/{rid}/cancel")
    assert r.status_code == 202
    monkeypatch.undo()
    d = (await client.get(f"/api/runs/{rid}")).json()
    assert d["state"] == "preparing" and d["cancel_requested"] is True and d["finished_at"] is None


async def test_events_release_db_session_while_streaming(client, session, session_factory, monkeypatch):
    monkeypatch.setattr(runs_router, "EVENTS_POLL_S", 0.02)
    rid = await make_run(client, session)
    app = client._transport.app
    real = app.state.sessionmaker
    open_count = {"n": 0}
    opened = {"n": 0}

    class Tracked:
        def __init__(self):
            self.cm = real()

        async def __aenter__(self):
            open_count["n"] += 1
            opened["n"] += 1
            return await self.cm.__aenter__()

        async def __aexit__(self, *a):
            try:
                return await self.cm.__aexit__(*a)
            finally:
                open_count["n"] -= 1

    app.state.sessionmaker = Tracked
    # Observe the open-session count exactly where the stream sleeps between polls (deterministic).
    at_sleep: list[int] = []
    opened_before_first_sleep: list[int] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(_delay):
        at_sleep.append(open_count["n"])
        opened_before_first_sleep.append(opened["n"])
        await set_state(session_factory, rid, state="done")  # let the stream finish on the next poll
        await real_sleep(0)

    monkeypatch.setattr(runs_router, "asyncio", SimpleNamespace(sleep=fake_sleep))
    r = await asyncio.wait_for(client.get(f"/api/runs/{rid}/events"), 5)
    assert r.status_code == 200
    assert at_sleep == [0]  # nothing (404-check session or per-poll session) held open across a sleep
    assert opened_before_first_sleep[0] >= 2  # 404-check session plus first poll session were used and closed
    assert open_count["n"] == 0


async def test_events_flush_last_line_without_newline(client, session, session_factory, monkeypatch):
    monkeypatch.setattr(runs_router, "EVENTS_POLL_S", 0.01)
    rid = await make_run(client, session)
    rd = RunDir((await client.get(f"/api/runs/{rid}")).json()["run_dir"])
    rd.ensure()
    (rd.logs_dir / "stdout.log").write_text("a\nlast", encoding="utf-8")
    await set_state(session_factory, rid, state="failed")
    events = parse_sse((await client.get(f"/api/runs/{rid}/events")).text)
    lines = [d["line"] for e, d in events if e == "log"]
    assert lines == ["a", "last"] and events[-1][0] == "state"


async def test_metrics_with_offset(client, session):
    rid = await make_run(client, session)
    rd = RunDir((await client.get(f"/api/runs/{rid}")).json()["run_dir"])
    rd.ensure()
    assert (await client.get(f"/api/runs/{rid}/metrics")).json() == {"items": [], "next_offset": 0}
    for i in range(2):
        append_metric(rd, Metric(step=i, split="train", name="loss", value=1.0 / (i + 1), ts=float(i)))
    body = (await client.get(f"/api/runs/{rid}/metrics")).json()
    assert [m["step"] for m in body["items"]] == [0, 1]
    append_metric(rd, Metric(step=2, split="train", name="loss", value=0.1, ts=2.0))
    more = (await client.get(f"/api/runs/{rid}/metrics?offset={body['next_offset']}")).json()
    assert [m["step"] for m in more["items"]] == [2]


async def test_logs_tail(client, session):
    rid = await make_run(client, session)
    assert (await client.get(f"/api/runs/{rid}/logs")).json() == {"stdout": [], "stderr": []}
    rd = RunDir((await client.get(f"/api/runs/{rid}")).json()["run_dir"])
    rd.ensure()
    (rd.logs_dir / "stdout.log").write_text("\n".join(f"l{i}" for i in range(10)) + "\n", encoding="utf-8")
    (rd.logs_dir / "stderr.log").write_bytes(b"ok\xff\n")
    body = (await client.get(f"/api/runs/{rid}/logs?tail=3")).json()
    assert body["stdout"] == ["l7", "l8", "l9"]
    assert body["stderr"] == ["ok�"]
    assert (await client.get("/api/runs/r-nope/logs")).status_code == 404


async def test_files_endpoint_blocks_traversal(client, session):
    rid = await make_run(client, session)
    rd = finish_files((await client.get(f"/api/runs/{rid}")).json()["run_dir"])
    rd.config_path.write_text("{}", encoding="utf-8")
    ok = await client.get(f"/api/runs/{rid}/files/export/model.bin")
    assert ok.status_code == 200 and ok.content == b"weights"
    assert (await client.get(f"/api/runs/{rid}/files/eval/results.json")).status_code == 200
    assert (await client.get(f"/api/runs/{rid}/files/../config.json")).status_code == 404
    assert (await client.get(f"/api/runs/{rid}/files/eval/../../config.json")).status_code == 404
    assert (await client.get(f"/api/runs/{rid}/files/config.json")).status_code == 404
    assert (await client.get(f"/api/runs/{rid}/files/eval")).status_code == 404
    assert (await client.get(f"/api/runs/{rid}/files/%2e%2e/config.json")).status_code == 404
    assert (await client.get("/api/runs/r-nope/files/eval/results.json")).status_code == 404


def parse_sse(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        ev, data = None, None
        for line in block.split("\n"):
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
        if ev:
            out.append((ev, json.loads(data)))
    return out


async def test_events_stream_ends_on_terminal_state(client, session, session_factory, monkeypatch):
    monkeypatch.setattr(runs_router, "EVENTS_POLL_S", 0.01)
    rid = await make_run(client, session)
    rd = RunDir((await client.get(f"/api/runs/{rid}")).json()["run_dir"])
    rd.ensure()
    write_status(rd, Status(phase="train", step=1, updated_at=1.0))
    for i in range(2):
        append_metric(rd, Metric(step=i, split="train", name="loss", value=0.5, ts=float(i)))
    (rd.logs_dir / "stdout.log").write_text("hello\nworld\n", encoding="utf-8")
    (rd.logs_dir / "stderr.log").write_text("warn\n", encoding="utf-8")
    await set_state(session_factory, rid, state="done")
    r = await client.get(f"/api/runs/{rid}/events")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    names = [e for e, _ in events]
    assert names[-1] == "state" and names.count("state") == 1
    assert events[-1][1] == {"state": "done"}
    assert names.count("metric") == 2 and names.count("status") == 1
    logs = [d for e, d in events if e == "log"]
    assert {"stream": "stdout", "line": "hello"} in logs and {"stream": "stderr", "line": "warn"} in logs
    assert len(logs) == 3


async def test_events_unknown_run_is_404(client):
    assert (await client.get("/api/runs/r-nope/events")).status_code == 404


def _big_log(path, n=4000):
    # 100-byte lines, 400 KB total: well over the 256 KiB first-read window
    lines = [f"line-{i:05d} " + "x" * 88 for i in range(n)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def _complete(line: str) -> bool:
    return len(line) == 99 and line.startswith("line-")


async def test_logs_tail_reads_only_the_last_256k_of_a_big_log(client, session):
    rid = await make_run(client, session)
    rd = RunDir((await client.get(f"/api/runs/{rid}")).json()["run_dir"])
    rd.ensure()
    lines = _big_log(rd.logs_dir / "stdout.log")
    out = (await client.get(f"/api/runs/{rid}/logs?tail=100000")).json()["stdout"]
    assert 2000 < len(out) < len(lines)
    assert all(_complete(ln) for ln in out)  # mid-file start: the partial first line is dropped
    assert out == lines[-len(out):]


async def test_events_first_log_read_is_bounded_metrics_replay_from_zero(client, session, session_factory, monkeypatch):
    monkeypatch.setattr(runs_router, "EVENTS_POLL_S", 0.01)
    rid = await make_run(client, session)
    rd = RunDir((await client.get(f"/api/runs/{rid}")).json()["run_dir"])
    rd.ensure()
    for i in range(3):
        append_metric(rd, Metric(step=i, split="train", name="loss", value=0.5, ts=float(i)))
    lines = _big_log(rd.logs_dir / "stdout.log")
    await set_state(session_factory, rid, state="done")
    events = parse_sse((await client.get(f"/api/runs/{rid}/events")).text)
    got = [d["line"] for e, d in events if e == "log" and d["stream"] == "stdout"]
    assert 2000 < len(got) < len(lines)
    assert all(_complete(ln) for ln in got)
    assert got == lines[-len(got):]
    assert [d["step"] for e, d in events if e == "metric"] == [0, 1, 2]


async def test_list_runs_limit(client, session):
    a = await make_run(client, session)
    b = await make_run(client, session)
    assert [i["id"] for i in (await client.get("/api/runs")).json()] == [b, a]
    assert [i["id"] for i in (await client.get("/api/runs?limit=1")).json()] == [b]
    assert (await client.get("/api/runs?limit=500")).status_code == 200
    assert (await client.get("/api/runs?limit=501")).status_code == 422
    assert (await client.get("/api/runs?limit=0")).status_code == 422


@pytest.fixture
async def run_with_dir(client, session):
    rid = await make_run(client, session)
    rd = RunDir((await client.get(f"/api/runs/{rid}")).json()["run_dir"])
    rd.ensure()
    return rid, rd


async def test_prepared_refs_empty_then_present(client, run_with_dir):
    rid, rd = run_with_dir
    assert (await client.get(f"/api/runs/{rid}/prepared")).json() == {"refs": []}
    write_prepared_refs(rd, [{"role": "train", "hash": "h1"}])
    assert (await client.get(f"/api/runs/{rid}/prepared")).json() == {"refs": [{"role": "train", "hash": "h1"}]}
    assert (await client.get("/api/runs/r-nope/prepared")).status_code == 404


async def test_eval_samples_paged(client, run_with_dir):
    rid, rd = run_with_dir
    for i in range(3):
        append_eval_sample(rd, {"i": i})
    body = (await client.get(f"/api/runs/{rid}/eval/samples?offset=1&limit=1")).json()
    assert body == {"items": [{"i": 1}], "total": 3}
    assert (await client.get(f"/api/runs/{rid}/eval/samples?offset=-1")).status_code == 422
    assert (await client.get(f"/api/runs/{rid}/eval/samples?limit=501")).status_code == 422
    assert (await client.get(f"/api/runs/{rid}/eval/samples?limit=0")).status_code == 422
    assert (await client.get("/api/runs/r-nope/eval/samples")).status_code == 404


async def test_eval_audio_serves_wav(client, run_with_dir):
    rid, rd = run_with_dir
    rd.eval_audio_dir.mkdir(parents=True, exist_ok=True)
    (rd.eval_audio_dir / "a-1.wav").write_bytes(b"RIFFdata")
    r = await client.get(f"/api/runs/{rid}/eval/audio/a-1.wav")
    assert r.status_code == 200 and r.content == b"RIFFdata" and r.headers["content-type"] == "audio/wav"
    assert (await client.get(f"/api/runs/{rid}/eval/audio/missing.wav")).status_code == 404


@pytest.mark.parametrize("name", ["..%2Fconfig.json", "../config.json", "a.txt", "link.wav"])
async def test_eval_audio_rejects_traversal(client, run_with_dir, name):
    rid, rd = run_with_dir
    rd.config_path.write_text("{}", encoding="utf-8")
    rd.eval_audio_dir.mkdir(parents=True, exist_ok=True)
    (rd.eval_audio_dir / "a.txt").write_bytes(b"x")
    if name == "link.wav":
        try:
            os.symlink(rd.config_path, rd.eval_audio_dir / "link.wav")
        except OSError:
            pytest.skip("symlinks not permitted")
    assert (await client.get(f"/api/runs/{rid}/eval/audio/{name}")).status_code == 404
