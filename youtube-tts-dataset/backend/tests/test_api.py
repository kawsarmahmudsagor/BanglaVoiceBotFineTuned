import threading
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from ytds.api import create_app
from ytds.config import Settings
from ytds.events import EventBus
from ytds.runner import video_dir, vocals_path
from ytds.store import Store

SR = 16000


class FakeRunner:
    def __init__(self):
        self.calls, self.llm = [], None
        self.in_flight: set[tuple[str, str]] = set()
        self.stop_on_skip = False  # True: Skip cancels the running stage at once

    def busy(self, dataset_id, video_id):
        return (dataset_id, video_id) in self.in_flight

    def start(self):
        self.calls.append(("start",))

    def stop(self, timeout=5.0):
        self.calls.append(("stop",))

    def enqueue(self, dataset_id):
        self.calls.append(("enqueue", dataset_id))

    def retry(self, dataset_id, video_id):
        self.calls.append(("retry", dataset_id, video_id))

    def skip(self, dataset_id, video_id):
        self.calls.append(("skip", dataset_id, video_id))
        if self.stop_on_skip:
            self.in_flight.discard((dataset_id, video_id))

    def wait_free(self, dataset_id, video_id, timeout):
        return not self.busy(dataset_id, video_id)

    def retry_llm(self, dataset_id):
        self.calls.append(("retry_llm", dataset_id))


class FakeLlm:
    ok = True

    def __init__(self, host, model, api_key, timeout=120.0):
        self.host, self.model = host, model

    def show(self):
        return FakeLlm.ok


def fake_extract(url, download=False):
    if "evil" in url:
        return {"id": "../../etc"}
    if "bad" in url:
        raise RuntimeError("Unsupported URL: " + url)
    if "playlist" in url:
        return {"entries": [{"id": "vid1", "url": "https://y/vid1"}, {"id": "vid2", "url": "https://y/vid2"}]}
    return {"id": url.rsplit("/", 1)[-1]}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr("ytds.pipeline.download._default_extract", fake_extract)
    monkeypatch.setattr("ytds.api.settings.OllamaClient", FakeLlm)
    settings = Settings(data_dir=tmp_path / "var", export_dir=tmp_path / "import")
    store, bus, runner = Store(tmp_path / "ytds.db"), EventBus(min_interval_s=0), FakeRunner()
    app = create_app(settings, runner=runner, store=store, bus=bus)
    with TestClient(app) as client:
        yield client, store, bus, runner, settings


def _dataset_with_clips(store, clips, vid="vid1"):
    ds = store.create_dataset("spk", "")
    store.add_video(ds, vid, f"https://y/{vid}")
    base = {"video_id": vid, "status": "auto", "text_final": "ক", "text_source": "stt"}
    store.replace_clips(ds, vid, [{**base, "id": f"{vid}_{i:04d}", "idx": i, **c} for i, c in enumerate(clips)])
    return ds


def test_create_dataset_dedupes_urls(env):
    client, store, _, runner, _ = env
    urls = "https://y/playlist\nhttps://y/vid1\n\nhttps://y/bad\nhttps://y/vid3\n"
    r = client.post("/api/datasets", json={"name": "spk", "urls": urls})
    assert r.status_code == 200
    body = r.json()
    assert body["added"] == 3 and body["duplicates"] == 1
    assert [e["url"] for e in body["errors"]] == ["https://y/bad"]
    ds = body["dataset_id"]
    assert [v["id"] for v in store.list_videos(ds)] == ["vid1", "vid2", "vid3"]
    assert ("enqueue", ds) in runner.calls

    again = client.post(f"/api/datasets/{ds}/videos", json={"urls": "https://y/vid2\nhttps://y/vid4"})
    assert again.json() == {"dataset_id": ds, "added": 1, "duplicates": 1, "errors": []}
    assert client.post("/api/datasets", json={"name": "spk", "urls": "https://y/x"}).status_code == 409

    detail = client.get(f"/api/datasets/{ds}").json()
    assert detail["name"] == "spk" and len(detail["videos"]) == 4
    assert detail["videos"][0]["stages"]["download"] is False
    assert client.get("/api/datasets").json()[0]["id"] == ds


def test_video_actions(env):
    client, store, _, runner, settings = env
    ds = _dataset_with_clips(store, [{"start_s": 0, "end_s": 3}])
    assert client.post(f"/api/datasets/{ds}/videos/vid1/retry").status_code == 200
    assert client.post(f"/api/datasets/{ds}/videos/vid1/skip").status_code == 200
    assert ("retry", ds, "vid1") in runner.calls and ("skip", ds, "vid1") in runner.calls
    assert client.post(f"/api/datasets/{ds}/videos/nope/skip").status_code == 404
    work = video_dir(settings.data_dir, ds, "vid1")
    work.mkdir(parents=True)
    assert client.delete(f"/api/datasets/{ds}/videos/vid1").status_code == 200
    assert store.list_videos(ds) == [] and store.clips(ds) == [] and not work.exists()


def test_review_patch_approve_and_edit(env):
    client, store, _, _, _ = env
    ds = _dataset_with_clips(store, [
        {"start_s": 0, "end_s": 3, "status": "flagged", "cer": 0.2},
        {"start_s": 3, "end_s": 6, "status": "needs_review", "text_source": "translated"},
        {"start_s": 6, "end_s": 9, "status": "pending_llm", "text_final": None},
        {"start_s": 9, "end_s": 12},
    ])
    queue = client.get(f"/api/datasets/{ds}/review").json()
    assert [c["reason"] for c in queue["clips"]] == ["cer", "translated", "pending_llm"]
    assert queue["remaining"] == 3 and queue["reviewed"] == 0
    only = client.get(f"/api/datasets/{ds}/review", params={"reason": "translated"}).json()["clips"]
    assert [c["id"] for c in only] == ["vid1_0001"]

    r = client.patch(f"/api/datasets/{ds}/clips/vid1_0000", json={"text_final": "নতুন", "status": "approved"})
    assert r.status_code == 200 and r.json()["status"] == "approved" and r.json()["text_final"] == "নতুন"
    r = client.patch(f"/api/datasets/{ds}/clips/vid1_0001", json={"text_final": "সম্পাদিত"})
    assert r.json()["status"] == "needs_review" and r.json()["text_final"] == "সম্পাদিত"
    client.patch(f"/api/datasets/{ds}/clips/vid1_0002", json={"status": "rejected"})
    assert client.patch(f"/api/datasets/{ds}/clips/vid1_0002", json={"status": "auto"}).status_code == 422
    assert client.patch(f"/api/datasets/{ds}/clips/nope", json={"status": "approved"}).status_code == 404

    queue = client.get(f"/api/datasets/{ds}/review").json()
    assert [c["id"] for c in queue["clips"]] == ["vid1_0001"]
    assert queue["remaining"] == 1 and queue["reviewed"] == 2
    approved = client.get(f"/api/datasets/{ds}/review", params={"status": "approved"}).json()["clips"]
    assert [(c["id"], c["reason"]) for c in approved] == [("vid1_0000", "cer")]


def test_export_409_with_unreviewed(env):
    client, store, _, _, _ = env
    ds = _dataset_with_clips(store, [
        {"start_s": 0, "end_s": 3},
        {"start_s": 3, "end_s": 6, "status": "needs_review", "text_source": "translated"},
    ])
    r = client.post(f"/api/datasets/{ds}/export", json={})
    assert r.status_code == 409 and r.json() == {"unreviewed": 1}

    ds2 = store.create_dataset("empty", "")
    r = client.post(f"/api/datasets/{ds2}/export", json={})
    assert r.status_code == 409 and r.json() == {"error": "nothing to export"}


def test_export_runs_in_background_and_reports_over_ws(env, monkeypatch):
    client, store, _, _, settings = env
    ds = _dataset_with_clips(store, [{"start_s": 0, "end_s": 3}])
    seen = {}
    release = threading.Event()

    def fake_export(store_, dataset_id, out_dir, on_progress, *, data_dir, skip_unreviewed_translated):
        seen.update(out_dir=out_dir, data_dir=data_dir, skip=skip_unreviewed_translated)
        release.wait(5)
        on_progress(1, 1)
        return {"clips": 1}

    monkeypatch.setattr("ytds.api.dataset.export_dataset", fake_export)
    with client.websocket_connect(f"/ws/datasets/{ds}") as ws:
        r = client.post(f"/api/datasets/{ds}/export", json={})
        assert r.status_code == 202
        assert client.post(f"/api/datasets/{ds}/export", json={}).status_code == 409  # already running
        release.set()
        assert ws.receive_json() == {"type": "progress", "dataset_id": ds, "stage": "export", "done": 1, "total": 1}
        assert ws.receive_json() == {"type": "stage_done", "dataset_id": ds, "stage": "export",
                                     "summary": {"clips": 1}}
    assert seen == {"out_dir": settings.export_dir / "spk", "data_dir": settings.data_dir, "skip": False}


def test_export_skip_unreviewed_translated(env, monkeypatch):
    client, store, _, _, _ = env
    ds = _dataset_with_clips(store, [
        {"start_s": 0, "end_s": 3},
        {"start_s": 3, "end_s": 6, "status": "needs_review", "text_source": "translated"},
    ])
    seen = {}

    def fake_export(store_, dataset_id, out_dir, on_progress, *, data_dir, skip_unreviewed_translated):
        seen["skip"] = skip_unreviewed_translated
        return {"clips": 1}

    monkeypatch.setattr("ytds.api.dataset.export_dataset", fake_export)
    r = client.post(f"/api/datasets/{ds}/export", json={"skip_unreviewed_translated": True})
    assert r.status_code == 202
    for _ in range(200):
        if seen and not client.app.state.exports:
            break
        time.sleep(0.01)
    assert seen == {"skip": True}

    # unreviewed translated clips never count as exportable
    store.update_clip(ds, "vid1_0000", status="rejected")
    r = client.post(f"/api/datasets/{ds}/export", json={"skip_unreviewed_translated": True})
    assert r.status_code == 409 and r.json() == {"error": "nothing to export"}


def test_retry_llm(env):
    client, store, _, runner, _ = env
    ds = store.create_dataset("spk", "")
    r = client.post(f"/api/datasets/{ds}/retry-llm")
    assert r.status_code == 202 and r.json() == {"ok": True}
    assert ("retry_llm", ds) in runner.calls
    assert client.post("/api/datasets/nope/retry-llm").status_code == 404


def test_stats(env):
    client, store, _, _, _ = env
    ds = _dataset_with_clips(store, [
        {"start_s": 0, "end_s": 36},
        {"start_s": 36, "end_s": 72},
        {"start_s": 72, "end_s": 75, "status": "dropped", "drop_reason": "dnsmos"},
        {"start_s": 75, "end_s": 78, "status": "rejected"},
        {"start_s": 78, "end_s": 80, "status": "flagged"},
    ])
    s = client.get(f"/api/datasets/{ds}/stats").json()
    assert s["clips_kept"] == 2 and s["hours_kept"] == pytest.approx(72 / 3600)
    assert s["dropped"]["dnsmos"] == 1 and s["dropped"]["rejected"] == 1 and s["dropped"]["unreviewed"] == 1
    assert s["videos"][0]["video_id"] == "vid1" and s["videos"][0]["clips_kept"] == 2
    assert sum(v["clips"] for v in s["splits"].values()) == 2
    assert sum(v["hours"] for v in s["splits"].values()) == pytest.approx(72 / 3600)


def test_ws_receives_published_event(env):
    client, _, bus, _, _ = env
    with client.websocket_connect("/ws/datasets/ds1") as ws:
        bus.publish({"type": "log", "dataset_id": "other", "message": "not mine"})
        bus.publish({"type": "stage_done", "dataset_id": "ds1", "video_id": "v", "stage": "download"})
        assert ws.receive_json() == {"type": "stage_done", "dataset_id": "ds1", "video_id": "v", "stage": "download"}


def test_clip_audio_supports_range(env):
    client, store, _, _, settings = env
    ds = _dataset_with_clips(store, [{"start_s": 0.5, "end_s": 1.5}])
    vocals = vocals_path(settings.data_dir, ds, "vid1")
    vocals.parent.mkdir(parents=True)
    sf.write(vocals, np.zeros(2 * SR, dtype=np.float32), SR, subtype="PCM_16")

    r = client.get(f"/api/datasets/{ds}/clips/vid1_0000/audio", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100
    cached = video_dir(settings.data_dir, ds, "vid1") / "clips" / "vid1_0000.wav"
    assert sf.info(cached).frames == SR
    full = client.get(f"/api/datasets/{ds}/clips/vid1_0000/audio")
    assert full.status_code == 200 and full.headers["content-type"] == "audio/wav"
    assert client.get(f"/api/datasets/{ds}/clips/nope/audio").status_code == 404


def test_health_shape(env):
    client = env[0]
    body = client.get("/api/health").json()
    assert set(body) == {"llm_ok", "cuda", "ffmpeg"} and all(isinstance(v, bool) for v in body.values())
    assert body["llm_ok"] is True


def test_settings_roundtrip_updates_live_llm(env, tmp_path):
    client, store, _, runner, settings = env
    got = client.get("/api/settings").json()
    assert got["llm_host"] == settings.ollama_host and got["llm_model"] == settings.llm_model
    assert got["thresholds"]["min_bn_ratio"] == 0.5 and got["cookies_path"] == ""

    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n")
    body = {"thresholds": {**got["thresholds"], "min_bn_ratio": 0.7}, "cookies_path": str(cookies),
            "llm_host": "http://h:1", "llm_model": "m2"}
    assert client.put("/api/settings", json=body).status_code == 200
    assert store.get_thresholds().min_bn_ratio == 0.7 and store.get_setting("cookies_path") == str(cookies)
    assert (runner.llm.host, runner.llm.model) == ("http://h:1", "m2")
    assert client.put("/api/settings", json={"cookies_path": str(tmp_path / "missing.txt")}).status_code == 400

    FakeLlm.ok = False
    try:
        r = client.post("/api/settings/llm-test").json()
    finally:
        FakeLlm.ok = True
    assert r["ok"] is False and "m2" in r["error"]

    ds = store.create_dataset("d", "")
    assert client.put(f"/api/datasets/{ds}", json={"speaker": "Rahim"}).status_code == 200
    assert store.get_dataset(ds)["speaker"] == "Rahim"


def test_lifespan_starts_and_stops_runner(tmp_path):
    runner = FakeRunner()
    app = create_app(Settings(data_dir=tmp_path), runner=runner, store=Store(tmp_path / "db"), bus=EventBus())
    with TestClient(app):
        assert runner.calls == [("start",)]
    assert runner.calls == [("start",), ("stop",)]


def test_default_runner_wiring_passes_cookies_and_store_llm(tmp_path, monkeypatch):
    from ytds.runner import JobRunner

    calls = []
    monkeypatch.setattr("ytds.pipeline.download.download", lambda *a, **k: calls.append(k) or {})
    store = Store(tmp_path / "ytds.db")
    store.set_setting("llm_model", "from-settings-page")
    app = create_app(Settings(data_dir=tmp_path), store=store, bus=EventBus())
    runner = app.state.runner
    assert isinstance(runner, JobRunner) and runner.llm.model == "from-settings-page"
    assert sorted(runner.gpu._loaders) == ["demucs", "ecapa", "model_a", "whisper_translate"]

    runner.stages.download("v", "u", tmp_path, lambda d, t: None)
    store.set_setting("cookies_path", str(tmp_path / "cookies.txt"))
    runner.stages.download("v", "u", tmp_path, lambda d, t: None)
    assert calls == [{"cookies": None}, {"cookies": tmp_path / "cookies.txt"}]


@pytest.mark.parametrize("name", ["", "   ", "a/b", "a\\b", "..", "a\\..\\b", "a\tb", "C:x", "/abs", "a<b", 'a"b', "a|b", "a?", "a*", "x."])
def test_create_dataset_rejects_unsafe_names(env, name):
    client, store = env[0], env[1]
    assert client.post("/api/datasets", json={"name": name, "urls": "https://y/vid1"}).status_code == 400
    assert store.list_datasets() == []


def test_invalid_video_id_is_a_per_line_error(env):
    client, store = env[0], env[1]
    body = client.post("/api/datasets", json={"name": "spk", "urls": "https://y/evil\nhttps://y/vid1"}).json()
    assert body["added"] == 1 and [e["url"] for e in body["errors"]] == ["https://y/evil"]
    assert [v["id"] for v in store.list_videos(body["dataset_id"])] == ["vid1"]


def test_export_refuses_unrelated_existing_out_dir(env, monkeypatch):
    client, store, _, _, settings = env
    ds = _dataset_with_clips(store, [{"start_s": 0, "end_s": 3}])
    monkeypatch.setattr("ytds.api.dataset.export_dataset", lambda *a, **k: {"clips": 1})
    target = settings.export_dir / "spk"
    target.mkdir(parents=True)
    (target / "keep.txt").write_text("mine")
    r = client.post(f"/api/datasets/{ds}/export", json={})
    assert r.status_code == 409 and r.json() == {"error": "out_dir exists and is not a previous export"}
    assert (target / "keep.txt").read_text() == "mine"

    other = settings.export_dir / "previous"
    other.mkdir()
    (other / "metadata.csv").write_text("file,text,speaker,split\n")
    assert client.post(f"/api/datasets/{ds}/export", json={"out_dir": str(other)}).status_code == 202


def test_video_actions_409_while_exporting(env):
    client, store = env[0], env[1]
    ds = _dataset_with_clips(store, [{"start_s": 0, "end_s": 3}])
    client.app.state.exports.add(ds)
    assert client.post(f"/api/datasets/{ds}/videos/vid1/retry").status_code == 409
    assert client.delete(f"/api/datasets/{ds}/videos/vid1").status_code == 409
    assert [v["id"] for v in store.list_videos(ds)] == ["vid1"]


def test_approve_with_empty_text_is_422(env):
    client, store = env[0], env[1]
    ds = _dataset_with_clips(store, [{"start_s": 0, "end_s": 3, "status": "pending_llm", "text_final": None}])
    url = f"/api/datasets/{ds}/clips/vid1_0000"
    assert client.patch(url, json={"status": "approved"}).status_code == 422
    assert client.patch(url, json={"status": "approved", "text_final": "  "}).status_code == 422
    assert store.get_clip(ds, "vid1_0000")["status"] == "pending_llm"
    assert client.patch(url, json={"status": "approved", "text_final": "ক"}).status_code == 200


def test_settings_lists_stage_models(env):
    client = env[0]
    client.put("/api/settings", json={"llm_model": "m3"})
    models = client.get("/api/settings").json()["models"]
    assert set(models) == {"separate", "vad", "dnsmos", "speaker", "model_a", "translate", "llm"}
    assert models["separate"] == "htdemucs" and models["llm"] == "m3"
    assert models["model_a"] == "SayedShaun/Qwen3-ASR-0.6B-Bengali"


@pytest.mark.parametrize("status", ["queued", "running", "skipped", "failed"])
def test_delete_refused_while_stage_will_not_stop(env, status):
    client, store, _, runner, _ = env
    ds = _dataset_with_clips(store, [{"start_s": 0, "end_s": 3}])
    store.update_video(ds, "vid1", status=status)
    runner.in_flight.add((ds, "vid1"))  # a stage that does not stop in time
    r = client.delete(f"/api/datasets/{ds}/videos/vid1")
    assert r.status_code == 409
    assert r.json()["detail"] == "video is still stopping; try again in a few seconds"
    assert ("skip", ds, "vid1") in runner.calls  # delete asked the stage to stop
    assert [v["id"] for v in store.list_videos(ds)] == ["vid1"]


def test_delete_stops_a_running_video_then_deletes(env):
    client, store, _, runner, _ = env
    ds = _dataset_with_clips(store, [{"start_s": 0, "end_s": 3}])
    store.update_video(ds, "vid1", status="running")
    runner.in_flight.add((ds, "vid1"))
    runner.stop_on_skip = True
    assert client.delete(f"/api/datasets/{ds}/videos/vid1").status_code == 200
    assert ("skip", ds, "vid1") in runner.calls and store.list_videos(ds) == []


def test_review_lists_auto_clips_on_request(env):
    client, store, _, _, _ = env
    ds = _dataset_with_clips(store, [
        {"start_s": 0, "end_s": 3, "status": "auto", "text_source": "stt", "cer": None, "text_final": "এক"},
        {"start_s": 3, "end_s": 6, "status": "flagged", "text_source": "stt", "cer": 0.2},
    ])
    queue = client.get(f"/api/datasets/{ds}/review").json()
    assert [c["id"] for c in queue["clips"]] == ["vid1_0001"]  # auto clips are not in the default queue
    auto = client.get(f"/api/datasets/{ds}/review", params={"status": "auto"}).json()["clips"]
    assert [(c["id"], c["reason"]) for c in auto] == [("vid1_0000", "auto")]
    r = client.patch(f"/api/datasets/{ds}/clips/vid1_0000", json={"status": "approved"})
    approved = client.get(f"/api/datasets/{ds}/review", params={"status": "approved"}).json()["clips"]
    assert r.status_code == 200 and [c["reason"] for c in approved] == ["auto"]
