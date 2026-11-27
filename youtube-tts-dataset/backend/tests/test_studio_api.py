import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from test_api import FakeLlm, FakeRunner, fake_extract
from ytds.api import create_app
from ytds.config import Settings
from ytds.events import EventBus
from ytds.runner import cut_dir, video_dir
from ytds.store import Store


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr("ytds.pipeline.download._default_extract", fake_extract)
    monkeypatch.setattr("ytds.api.settings.OllamaClient", FakeLlm)
    (tmp_path / "model_a").mkdir()
    (tmp_path / "model_b").mkdir()
    settings = Settings(data_dir=tmp_path / "var", export_dir=tmp_path / "import",
                        model_a_path=str(tmp_path / "model_a"), model_b_path=str(tmp_path / "model_b"))
    store, bus, runner = Store(tmp_path / "ytds.db"), EventBus(min_interval_s=0), FakeRunner()
    app = create_app(settings, runner=runner, store=store, bus=bus)
    with TestClient(app) as client:
        yield client, store, runner, settings


def _ds(store):
    return store.create_dataset("spk", "")


def _video(store, ds, vid):
    return next(v for v in store.list_videos(ds) if v["id"] == vid)


def _awaiting(store, ds, vid="vid1", duration=60.0):
    store.add_video(ds, vid, f"https://y/{vid}", origin="studio", status="awaiting_cut")
    store.update_video(ds, vid, duration_s=duration)
    store.mark_stage(ds, vid, "download", True)


# --- add by URL
def test_add_url_creates_fetching_studio_video(env):
    client, store, runner, _ = env
    ds = _ds(store)
    r = client.post(f"/api/datasets/{ds}/studio/videos", json={"url": "https://y/vid1"})
    assert r.status_code == 200 and r.json() == {"video_id": "vid1"}
    v = _video(store, ds, "vid1")
    assert (v["status"], v["origin"], v["source"]) == ("fetching", "studio", "youtube")
    assert ("enqueue", ds) in runner.calls

    dup = client.post(f"/api/datasets/{ds}/studio/videos", json={"url": "https://y/vid1"})
    assert dup.status_code == 409 and dup.json()["detail"]["video_id"] == "vid1"


def test_add_url_rejects_playlist_and_bad_url(env):
    client, store, _, _ = env
    ds = _ds(store)
    r = client.post(f"/api/datasets/{ds}/studio/videos", json={"url": "https://y/playlist"})
    assert r.status_code == 400 and "playlist" in r.json()["detail"]["error"]
    r = client.post(f"/api/datasets/{ds}/studio/videos", json={"url": "https://y/bad"})
    assert r.status_code == 400 and "Unsupported URL" in r.json()["detail"]["error"]
    assert store.list_videos(ds) == []
    assert client.post("/api/datasets/nope/studio/videos", json={"url": "https://y/vid1"}).status_code == 404


# --- upload (Review Focus 5)
DATA = bytes(range(256)) * 10000  # 2.56 MB: three 1 MB chunks
LOCAL_ID = "local_" + hashlib.sha1(DATA).hexdigest()[:11]


def _upload(client, ds, data=DATA, name="My Talk.MP4"):
    return client.post(f"/api/datasets/{ds}/studio/upload", params={"filename": name}, content=data,
                       headers={"Content-Type": "application/octet-stream"})


def test_upload_stores_file_and_dedupes(env):
    client, store, runner, settings = env
    ds = _ds(store)
    r = _upload(client, ds)
    assert r.status_code == 200 and r.json() == {"video_id": LOCAL_ID}
    v = _video(store, ds, LOCAL_ID)
    assert (v["status"], v["origin"], v["source"], v["title"]) == ("fetching", "studio", "local", "My Talk.MP4")
    stored = video_dir(settings.data_dir, ds, LOCAL_ID) / "download" / "upload.mp4"
    assert stored.read_bytes() == DATA
    assert ("enqueue", ds) in runner.calls

    again = _upload(client, ds, name="copy.mp4")
    assert again.status_code == 409 and again.json()["detail"]["video_id"] == LOCAL_ID
    assert len(store.list_videos(ds)) == 1
    files = [p for p in (settings.data_dir / "videos" / ds).rglob("*") if p.is_file()]
    assert files == [stored]


def test_upload_failing_mid_stream_leaves_nothing(env, monkeypatch):
    client, store, _, settings = env
    ds = _ds(store)
    calls = []

    def flaky(f, chunk):
        calls.append(len(chunk))
        if len(calls) > 1:
            raise OSError("disk full")
        f.write(chunk)

    monkeypatch.setattr("ytds.api.studio._write_chunk", flaky)
    r = _upload(client, ds)
    assert r.status_code in (400, 500)
    assert len(calls) == 2
    assert store.list_videos(ds) == []
    root = settings.data_dir / "videos" / ds
    assert not root.exists() or [p for p in root.rglob("*")] == []


def test_upload_needs_filename_and_bytes(env):
    client, store, _, settings = env
    ds = _ds(store)
    assert client.post(f"/api/datasets/{ds}/studio/upload", content=b"abc").status_code == 400
    assert _upload(client, ds, name="  ").status_code == 400
    assert _upload(client, ds, data=b"").status_code == 400
    assert store.list_videos(ds) == []
    root = settings.data_dir / "videos" / ds
    assert not root.exists() or [p for p in root.rglob("*")] == []
    assert _upload(client, "nope").status_code == 404


def test_upload_name_and_suffix_from_filename(env):
    client, store, _, settings = env
    ds = _ds(store)
    vid = _upload(client, ds, data=b"abc", name=r"C:\clips\Lecture One.WebM").json()["video_id"]
    assert _video(store, ds, vid)["title"] == "Lecture One.WebM"
    assert (video_dir(settings.data_dir, ds, vid) / "download" / "upload.webm").read_bytes() == b"abc"
    vid = _upload(client, ds, data=b"def", name="noext").json()["video_id"]
    assert (video_dir(settings.data_dir, ds, vid) / "download" / "upload").read_bytes() == b"def"


# --- list
def test_list_studio_videos(env):
    client, store, _, settings = env
    ds = _ds(store)
    store.add_video(ds, "dash", "https://y/dash")  # dashboard video: not listed
    _awaiting(store, ds, "vid1")
    store.update_video(ds, "vid1", title="t", media="download/video.mp4", cuts=json.dumps([[0, 10]]))
    media = video_dir(settings.data_dir, ds, "vid1") / "download" / "video.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"x")
    store.add_video(ds, "vid2", "https://y/vid2", origin="studio", status="fetching")
    store.update_video(ds, "vid2", media="download/gone.mp4")  # file missing
    base = {"video_id": "vid1", "text_source": "stt"}
    store.replace_clips(ds, "vid1", [
        {**base, "id": "vid1_0000", "idx": 0, "status": "auto"},
        {**base, "id": "vid1_0001", "idx": 1, "status": "flagged"},
        {**base, "id": "vid1_0002", "idx": 2, "status": "auto", "flag": "disagree"},
        {**base, "id": "vid1_0003", "idx": 3, "status": "dropped"},
    ])
    out = client.get(f"/api/datasets/{ds}/studio/videos").json()
    assert [v["id"] for v in out] == ["vid1", "vid2"]
    v1, v2 = out
    assert v1["media_ready"] is True and v2["media_ready"] is False
    assert v1["cuts"] == [[0, 10]] and v2["cuts"] is None
    assert v1["stages"]["download"] is True and v1["stages"]["cut"] is False
    assert v1["clips"] == {"total": 4, "kept": 3, "flagged": 2, "dropped": 1}
    assert {k for k in v1} >= {"id", "title", "status", "error", "warning", "duration_s", "media_ready", "cuts",
                               "stages", "clips"}


# --- media
def test_media_range_request(env):
    client, store, _, settings = env
    ds = _ds(store)
    _awaiting(store, ds)
    url = f"/api/datasets/{ds}/videos/vid1/media"
    assert client.get(url).status_code == 404  # media NULL
    store.update_video(ds, "vid1", media="download/video.mp4")
    assert client.get(url).status_code == 404  # file missing
    f = video_dir(settings.data_dir, ds, "vid1") / "download" / "video.mp4"
    f.parent.mkdir(parents=True)
    f.write_bytes(bytes(range(100)))
    r = client.get(url, headers={"Range": "bytes=0-9"})
    assert r.status_code == 206 and r.content == bytes(range(10))
    assert client.get(url).content == bytes(range(100))


def test_media_path_cannot_escape_video_dir(env):
    client, store, _, settings = env
    ds = _ds(store)
    _awaiting(store, ds)
    (settings.data_dir / "secret.txt").parent.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "secret.txt").write_text("s")
    store.update_video(ds, "vid1", media="../../../secret.txt")
    assert client.get(f"/api/datasets/{ds}/videos/vid1/media").status_code == 404


# --- start (Review Focus 2)
def _start(client, ds, cuts, vid="vid1"):
    return client.post(f"/api/datasets/{ds}/videos/{vid}/start", json={"cuts": cuts})


def test_start_only_in_awaiting_cut(env):
    client, store, runner, _ = env
    ds = _ds(store)
    store.add_video(ds, "vid1", "https://y/vid1", origin="studio", status="queued")
    assert _start(client, ds, []).status_code == 409
    assert _start(client, ds, [], vid="nope").status_code == 404
    assert not [c for c in runner.calls if c[0] == "start_video"]


@pytest.mark.parametrize("cuts", [[[5, 3]], [[3, 3]], [[70, 80]], [[0, 0.5]], [["a", 3]], [[1]], [[1, 2, 3]],
                                  [[True, 3]], "x", [5]])
def test_start_rejects_bad_cuts(env, cuts):
    client, store, runner, _ = env
    ds = _ds(store)
    _awaiting(store, ds)
    r = _start(client, ds, cuts)
    assert r.status_code == 422
    assert not [c for c in runner.calls if c[0] == "start_video"]
    assert _video(store, ds, "vid1")["status"] == "awaiting_cut"


@pytest.mark.parametrize("raw", [b'{"cuts": [[NaN, 5]]}', b'{"cuts": [[0, Infinity]]}', b'{"cuts": [[-Infinity, 5]]}'])
def test_start_rejects_non_finite(env, raw):
    client, store, runner, _ = env
    ds = _ds(store)
    _awaiting(store, ds)
    r = client.post(f"/api/datasets/{ds}/videos/vid1/start", content=raw,
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert not [c for c in runner.calls if c[0] == "start_video"]


def test_bad_cuts_error_message(env):
    client, store, _, _ = env
    ds = _ds(store)
    _awaiting(store, ds)
    r = _start(client, ds, [[5, 3]])
    assert "before end" in r.json()["detail"]["error"]


def test_start_normalises_ranges(env):
    client, store, runner, _ = env
    ds = _ds(store)
    _awaiting(store, ds, duration=50.0)
    r = _start(client, ds, [[40, 70], [15, 30], [10, 20], [3, 3.5], [45.2, 45.5]])
    assert r.status_code == 200
    assert runner.calls[-1] == ("start_video", ds, "vid1", [[10.0, 30.0], [40.0, 50.0]])


def test_start_empty_means_whole_video(env):
    client, store, runner, _ = env
    ds = _ds(store)
    _awaiting(store, ds)
    assert _start(client, ds, []).status_code == 200
    assert runner.calls[-1] == ("start_video", ds, "vid1", [])
    store.update_video(ds, "vid1", status="awaiting_cut")
    assert client.post(f"/api/datasets/{ds}/videos/vid1/start", json={}).status_code == 200
    assert runner.calls[-1] == ("start_video", ds, "vid1", [])


def test_start_needs_both_models(env, tmp_path):
    client, store, runner, _ = env
    ds = _ds(store)
    _awaiting(store, ds)
    (tmp_path / "model_b").rmdir()
    r = _start(client, ds, [[10, 20]])
    assert r.status_code == 422 and str(tmp_path / "model_b") in r.json()["detail"]["error"]
    assert not [c for c in runner.calls if c[0] == "start_video"]


# --- clips
def test_clips_lists_every_status_with_original_times(env):
    client, store, _, settings = env
    ds = _ds(store)
    _awaiting(store, ds)
    store.update_video(ds, "vid1", status="done")
    base = {"video_id": "vid1", "text_source": "stt"}
    store.replace_clips(ds, "vid1", [
        {**base, "id": "vid1_0000", "idx": 0, "status": "auto", "start_s": 1.0, "end_s": 16.0, "text_final": "ক"},
        {**base, "id": "vid1_0001", "idx": 1, "status": "dropped", "start_s": 22.0, "end_s": 40.0,
         "drop_reason": "length"},
        {**base, "id": "vid1_0002", "idx": 2, "status": "flagged", "start_s": 23.0, "end_s": 39.0,
         "flag": "disagree", "range_idx": 1},
        {**base, "id": "vid1_0003", "idx": 3, "status": None, "start_s": 2.0, "end_s": 3.0, "text_final": None},
    ])
    url = f"/api/datasets/{ds}/videos/vid1/clips"
    plain = {c["id"]: c for c in client.get(url).json()}
    assert plain["vid1_0001"]["orig_start_s"] == 22.0 and plain["vid1_0001"]["orig_end_s"] == 40.0  # no map

    cmap = [{"range_idx": 0, "cut_start_s": 0.0, "cut_end_s": 20.0, "orig_start_s": 100.0},
            {"range_idx": 1, "cut_start_s": 22.0, "cut_end_s": 42.0, "orig_start_s": 300.0}]
    d = cut_dir(settings.data_dir, ds, "vid1")
    d.mkdir(parents=True)
    (d / "map.json").write_text(json.dumps(cmap), encoding="utf-8")
    clips = {c["id"]: c for c in client.get(url).json()}
    assert list(clips) == ["vid1_0000", "vid1_0001", "vid1_0002", "vid1_0003"]
    assert clips["vid1_0000"]["orig_start_s"] == 101.0 and clips["vid1_0000"]["orig_end_s"] == 116.0
    assert clips["vid1_0002"]["orig_start_s"] == 301.0 and clips["vid1_0002"]["orig_end_s"] == 317.0
    assert clips["vid1_0001"]["status"] == "dropped"
    assert clips["vid1_0000"]["reason"] == "auto"
    assert clips["vid1_0002"]["reason"] == "disagree" and clips["vid1_0002"]["flag"] == "disagree"
    assert clips["vid1_0003"]["reason"] is None
    assert client.get(f"/api/datasets/{ds}/videos/nope/clips").status_code == 404


# --- review swap
def test_patch_text_pick_swaps_text(env):
    client, store, _, _ = env
    ds = _ds(store)
    store.add_video(ds, "vid1", "https://y/vid1")
    store.replace_clips(ds, "vid1", [{"id": "vid1_0000", "video_id": "vid1", "idx": 0, "status": "flagged",
                                      "text_a": "এ", "text_b": "বি", "text_pick": "b", "text_final": "বি।",
                                      "text_source": "stt"}])
    r = client.patch(f"/api/datasets/{ds}/clips/vid1_0000", json={"text_pick": "a"})
    assert r.status_code == 200
    c = store.get_clip(ds, "vid1_0000")
    assert (c["text_final"], c["text_pick"], c["status"]) == ("এ", "a", "flagged")
    r = client.patch(f"/api/datasets/{ds}/clips/vid1_0000", json={"text_pick": "b", "text_final": "নিজে"})
    c = store.get_clip(ds, "vid1_0000")
    assert (c["text_final"], c["text_pick"]) == ("নিজে", "b")
    assert client.patch(f"/api/datasets/{ds}/clips/vid1_0000", json={"text_pick": "c"}).status_code == 422


# --- SPA route
def test_studio_route_serves_index(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>studio app</html>", encoding="utf-8")
    monkeypatch.setattr("ytds.api.FRONTEND_DIST", dist)
    monkeypatch.setattr("ytds.api.settings.OllamaClient", FakeLlm)
    settings = Settings(data_dir=tmp_path / "var", export_dir=tmp_path / "import")
    app = create_app(settings, runner=FakeRunner(), store=Store(tmp_path / "db.sqlite"),
                     bus=EventBus(min_interval_s=0))
    with TestClient(app) as client:
        for path in ("/studio", "/studio/", "/"):
            r = client.get(path)
            assert r.status_code == 200 and "studio app" in r.text and r.headers["content-type"].startswith("text/html")
        assert client.get("/api/datasets").status_code == 200


class _StreamRequest:
    """Just enough of a Starlette request for `_receive`: a chunked body stream that can disconnect."""

    def __init__(self, body: bytes, step: int, fail_after: int):
        self._body, self._step, self._fail_after = body, step, fail_after

    async def stream(self):
        from starlette.requests import ClientDisconnect

        for n, i in enumerate(range(0, len(self._body), self._step)):
            if n >= self._fail_after:
                raise ClientDisconnect()
            yield self._body[i:i + self._step]


def test_receive_closes_file_on_disconnect(tmp_path):
    import asyncio
    import shutil

    from starlette.requests import ClientDisconnect

    from ytds.api.studio import _receive

    with pytest.raises(ClientDisconnect):
        asyncio.run(_receive(_StreamRequest(DATA, 1 << 16, fail_after=20), tmp_path / "dl" / "upload.mp4"))
    shutil.rmtree(tmp_path / "dl")  # fails on Windows if the file were still open
    assert not (tmp_path / "dl").exists()
