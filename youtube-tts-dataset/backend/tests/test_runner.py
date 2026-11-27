import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ytds.config import Settings
from ytds.pipeline import STAGES
from ytds.runner import GPU_STAGES, JobRunner, StageFns, cut_dir, vocals_path
from ytds.store import Store

IDLE_S = 10
BN = "আমি বাংলায় কথা বলছি এখানে"
CMAP = [{"range_idx": 0, "cut_start_s": 0.0, "cut_end_s": 60.0, "orig_start_s": 0.0}]


class FakeBus:
    def __init__(self):
        self.events, self._lock = [], threading.Lock()

    def publish(self, event):
        with self._lock:
            self.events.append(dict(event))

    def of(self, type_):
        with self._lock:
            return [e for e in self.events if e["type"] == type_]


def stages_for(bus, vid):
    """Stages that ran for `vid`, in order (done, failed or cancelled)."""
    with bus._lock:
        return [e["stage"] for e in bus.events
                if e["type"] in ("stage_done", "stage_failed", "stage_cancelled") and e.get("video_id") == vid]


class FakeModel:
    def __init__(self, option_id):
        self.option_id = option_id


class FakeGpu:
    """Caches one model like GpuWorker; model_a/model_b carry `.option_id` read when they load."""

    def __init__(self, order, options=None):
        self.used, self.order = [], order
        self.options = options or {}
        self.loaded, self._model = None, None

    def use(self, name):
        self.used.append(name)
        self.order.append(("use", name))
        if self.loaded != name:
            if name in ("model_a", "model_b"):
                self._model = FakeModel(self.options.get(name, lambda: f"fake-{name[-1]}")())
            else:
                self._model = name
            self.loaded = name
        return self._model

    def release(self):
        self.order.append(("release",))
        self.loaded = self._model = None

    def unload(self, name):
        if self.loaded == name:
            self.loaded = self._model = None


def _vid(path) -> str:
    return Path(path).parent.parent.name  # <dataset>/<video>/<stage>/<file>


def default_texts(vid, i):
    t = f"{vid} {BN}" + (" end" if i % 3 == 2 else "")
    return t, t


class Fakes:
    """Stage fakes with the real stage signatures; `log` records (stage, video_id) in call order."""

    def __init__(self, n_pieces=12, piece_s=4.0, gap_s=0.5, fail=None, texts=None):
        self.log, self._lock = [], threading.Lock()
        self.n_pieces, self.piece_s, self.gap_s = n_pieces, piece_s, gap_s
        self.fail = fail or {}  # (stage, video_id) -> exception
        self.texts = texts or default_texts  # (vid, piece idx) -> (text_a, text_b)
        self.llm_up = True
        self.llm_down_if = None  # text -> bool: the LLM fails for these texts
        self.cuts_seen = {}
        self.order = []  # shared with the FakeGpu: use / release / llm / llm_unload in call order

    def _rec(self, stage, vid):
        with self._lock:
            self.log.append((stage, vid))
        exc = self.fail.get((stage, vid))
        if exc:
            raise exc

    def calls_for(self, vid):
        with self._lock:
            return [s for s, v in self.log if v == vid]

    def download(self, video_id, url, out_dir, on_progress):
        self._rec("download", video_id)
        on_progress(50, 100)
        on_progress(100, 100)
        return {"title": f"title {video_id}", "duration_s": 60.0}

    def cut(self, audio_wav, out_dir, cuts):
        vid = Path(out_dir).parent.name
        self._rec("cut", vid)
        self.cuts_seen[vid] = cuts
        assert Path(audio_wav).parent.name == "download"
        return [dict(m) for m in CMAP]

    def separate(self, audio_wav, out_wav, model, segment_s, on_progress):
        assert model == "demucs"
        assert Path(audio_wav).parent.name == "cut"
        self._rec("separate", _vid(out_wav))
        on_progress(1, 1)

    def pieces(self, vocals_wav, cmap, th):
        assert cmap == CMAP
        self._rec("pieces", _vid(vocals_wav))
        step = self.piece_s + self.gap_s
        return [{"idx": i, "range_idx": 0, "start_s": i * step, "end_s": i * step + self.piece_s}
                for i in range(self.n_pieces)]

    def transcribe_a(self, pieces, vocals_wav, model, on_progress):
        vid = _vid(vocals_wav)
        self._rec("transcribe_a", vid)
        on_progress(len(pieces), len(pieces))
        return [self.texts(vid, p["idx"])[0] for p in pieces]

    def transcribe_b(self, pieces, vocals_wav, model, on_progress):
        vid = _vid(vocals_wav)
        self._rec("transcribe_b", vid)
        on_progress(len(pieces), len(pieces))
        return [{"text": self.texts(vid, p["idx"])[1], "tokens": 10, "compression": 1.2, "retry_failed": False}
                for p in pieces]

    def quality(self, vocals_wav, clips, th, scorer=None, embedder=None):
        assert embedder == "ecapa"
        self._rec("quality", _vid(vocals_wav))
        for c in clips:
            c.update(dnsmos=3.5, speaker_sim=0.9, drop_reason=None)
        return clips

    def punctuate(self, text, llm):
        self._rec("punctuate", text.split()[0] if text.split() else "")
        self.order.append(("llm",))
        if not self.llm_up or (self.llm_down_if and self.llm_down_if(text)):
            return text, True
        return (text + "।" if "end" in text else text), False

    def llm_unload(self, llm):
        self.order.append(("llm_unload",))

    def stage_fns(self):
        return StageFns(download=self.download, cut=self.cut, separate=self.separate, pieces=self.pieces,
                        transcribe_a=self.transcribe_a, transcribe_b=self.transcribe_b, quality=self.quality,
                        punctuate=self.punctuate, llm_unload=self.llm_unload)


@pytest.fixture
def env(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    bus = FakeBus()
    settings = Settings(data_dir=tmp_path / "var")
    runners = []

    def make(fakes, gpu=None):
        r = JobRunner(store, bus, gpu or FakeGpu(fakes.order), llm=object(), settings=settings,
                      stages=fakes.stage_fns())
        runners.append(r)
        return r

    yield store, bus, settings, make
    for r in runners:
        r.stop()


def _dataset(store, *vids, **kw):
    ds = store.create_dataset("d", "spk")
    for v in vids:
        store.add_video(ds, v, f"https://youtu.be/{v}", **kw)
    return ds


def _video(store, ds, vid):
    return next(v for v in store.list_videos(ds) if v["id"] == vid)


def _run(runner, ds):
    runner.start()
    runner.enqueue(ds)
    assert runner.wait_idle(IDLE_S)


def _kept(c):
    return c["text_b"] if c["text_pick"] == "b" else c["text_a"]


def test_stage_constants():
    assert STAGES == ("download", "cut", "separate", "pieces", "transcribe_a", "transcribe_b",
                      "punctuate_pieces", "group", "quality", "select", "punctuate")
    assert GPU_STAGES == ("separate", "transcribe_a", "transcribe_b", "punctuate_pieces", "quality", "punctuate")


def test_runs_all_stages_in_order(env):
    store, bus, settings, make = env
    fakes = Fakes()
    ds = _dataset(store, "v1")
    _run(make(fakes), ds)

    assert stages_for(bus, "v1") == list(STAGES)
    v = _video(store, ds, "v1")
    assert v["status"] == "done" and v["warning"] is None and v["error"] is None
    assert v["title"] == "title v1"
    assert (v["model_a"], v["model_b"]) == ("fake-a", "fake-b")
    assert all(store.stage_done(ds, "v1", s) for s in STAGES)
    assert json.loads((cut_dir(settings.data_dir, ds, "v1") / "map.json").read_text(encoding="utf-8")) == CMAP

    pieces = store.pieces(ds, "v1")
    assert len(pieces) == 12
    assert all(p["text_a"] == p["text_b"] and p["b_tokens"] == 10 and p["b_compression"] == 1.2 for p in pieces)
    assert all(p["broken"] == 0 and p["no_punct"] == 0 for p in pieces)
    assert [p["text"].endswith("।") for p in pieces] == [i % 3 == 2 for i in range(12)]

    th = store.get_thresholds()
    clips = store.clips(ds, video_id="v1")
    assert [c["id"] for c in clips] == [f"v1_{i:04d}" for i in range(len(clips))]
    live = [c for c in clips if c["status"] != "dropped"]
    assert len(live) == 4
    for c in live:
        dur = c["end_s"] - c["start_s"]
        assert dur <= th.clip_max_s + 0.2
        assert dur >= th.clip_min_s or (dur >= th.tail_min_s and c["text_final"].endswith("।"))
        assert c["range_idx"] == 0 and c["flag"] is None and c["status"] == "auto"
        assert c["text_pick"] == "b" and c["cer"] == 0.0 and c["text_source"] == "stt"
        assert c["text_final"] == _kept(c) + "।" and c["no_punct"] == 0
        assert c["dnsmos"] == 3.5
    assert [e["stage"] for e in bus.of("stage_done")] == list(STAGES)
    assert len(bus.of("video_done")) == 1
    progress = bus.of("progress")
    assert {"dataset_id": ds, "video_id": "v1", "stage": "download", "done": 50, "total": 100}.items() \
        <= progress[0].items()
    assert "eta_s" in progress[0]


def test_dashboard_video_runs_straight_through(env):
    store, bus, settings, make = env
    fakes = Fakes()
    ds = _dataset(store, "v1")
    _run(make(fakes), ds)
    assert fakes.cuts_seen == {"v1": None}
    assert _video(store, ds, "v1")["status"] == "done"


def test_studio_video_stops_at_awaiting_cut(env):
    store, bus, settings, make = env
    fakes = Fakes()
    ds = _dataset(store, "v1", origin="studio", status="fetching")
    seen = []
    orig = fakes.download

    def download(video_id, url, out_dir, on_progress):
        seen.append(_video(store, ds, video_id)["status"])
        return orig(video_id, url, out_dir, on_progress)

    fakes.download = download
    runner = make(fakes)
    runner.stages.download = download
    _run(runner, ds)
    assert seen == ["fetching"]  # the status stays fetching while it downloads
    assert _video(store, ds, "v1")["status"] == "awaiting_cut"
    assert fakes.calls_for("v1") == ["download"]

    runner.start_video(ds, "v1", [[0, 30]])
    assert runner.wait_idle(IDLE_S)
    v = _video(store, ds, "v1")
    assert v["status"] == "done" and json.loads(v["cuts"]) == [[0, 30]]
    assert fakes.cuts_seen == {"v1": [[0, 30]]}
    assert stages_for(bus, "v1") == list(STAGES)


def test_restart_keeps_awaiting_cut_and_resumes_fetching(env):
    store, bus, settings, make = env
    fakes = Fakes()
    ds = _dataset(store)
    store.add_video(ds, "v1", "u1", origin="studio", status="awaiting_cut")
    store.mark_stage(ds, "v1", "download", True)
    store.add_video(ds, "v2", "u2", origin="studio", status="fetching")  # download never finished
    store.add_video(ds, "v3", "u3", origin="studio", status="fetching")  # died after marking download done
    store.mark_stage(ds, "v3", "download", True)

    runner = make(fakes)
    runner.start()  # no enqueue: start alone resumes
    assert runner.wait_idle(IDLE_S)
    assert _video(store, ds, "v1")["status"] == "awaiting_cut"
    assert _video(store, ds, "v2")["status"] == "awaiting_cut"
    assert _video(store, ds, "v3")["status"] == "awaiting_cut"
    assert fakes.log == [("download", "v2")]


def test_retry_studio_download_returns_to_fetching(env):
    store, bus, settings, make = env
    fakes = Fakes(fail={("download", "v1"): RuntimeError("yt-dlp: 403")})
    ds = _dataset(store, "v1", origin="studio", status="fetching")
    runner = make(fakes)
    _run(runner, ds)
    assert _video(store, ds, "v1")["status"] == "failed"

    fakes.fail = {}
    seen = []
    orig = fakes.download

    def download(video_id, url, out_dir, on_progress):
        seen.append(_video(store, ds, video_id)["status"])
        return orig(video_id, url, out_dir, on_progress)

    runner.stages.download = download
    runner.retry(ds, "v1")
    assert runner.wait_idle(IDLE_S)
    assert seen == ["fetching"]
    assert _video(store, ds, "v1")["status"] == "awaiting_cut"


def test_resume_skips_done_stages(env):
    store, bus, settings, make = env
    fakes = Fakes()
    ds = _dataset(store, "v1")
    store.mark_stage(ds, "v1", "download", True)
    _run(make(fakes), ds)

    assert stages_for(bus, "v1") == list(STAGES[1:])
    assert "download" not in fakes.calls_for("v1")
    assert _video(store, ds, "v1")["status"] == "done"


def test_failure_isolated_to_video(env):
    store, bus, settings, make = env
    fakes = Fakes(fail={("pieces", "v1"): RuntimeError("vad exploded")})
    ds = _dataset(store, "v1", "v2")
    _run(make(fakes), ds)

    v1, v2 = _video(store, ds, "v1"), _video(store, ds, "v2")
    assert v1["status"] == "failed" and v1["error"] == "vad exploded"
    assert v2["status"] == "done"
    assert stages_for(bus, "v1") == ["download", "cut", "separate", "pieces"]
    assert not store.stage_done(ds, "v1", "pieces")
    failed = bus.of("stage_failed")
    assert len(failed) == 1
    assert failed[0]["video_id"] == "v1" and failed[0]["stage"] == "pieces"
    assert failed[0]["error"] == "vad exploded"


def test_gpu_stage_major_across_videos(env):
    store, bus, settings, make = env
    fakes = Fakes()
    ds = _dataset(store, "v1", "v2")
    _run(make(fakes), ds)

    seq = [s for s, _ in fakes.log]
    assert seq.count("separate") == 2 and seq.count("transcribe_a") == 2
    assert max(i for i, s in enumerate(seq) if s == "separate") < min(i for i, s in enumerate(seq) if s == "transcribe_a")
    assert max(i for i, s in enumerate(seq) if s == "transcribe_a") < min(i for i, s in enumerate(seq) if s == "transcribe_b")
    assert all(_video(store, ds, v)["status"] == "done" for v in ("v1", "v2"))


def test_llm_is_a_gpu_tenant(env):
    """release -> LLM work -> llm_unload -> next model load, for every LLM stage."""
    store, bus, settings, make = env
    fakes = Fakes()
    ds = _dataset(store, "v1", "v2")
    _run(make(fakes), ds)

    order = fakes.order
    assert ("llm",) in order and ("llm_unload",) in order
    state = "free"
    for ev in order:
        if ev[0] == "release":
            state = "free"
        elif ev[0] == "llm":
            assert state in ("free", "llm"), f"LLM ran while a model was loaded: {order}"
            state = "llm"
        elif ev[0] == "llm_unload":
            state = "free"
        elif ev[0] == "use":
            assert state != "llm", f"model loaded before the LLM was unloaded: {order}"
            state = "model"
    assert state != "llm"


def test_no_speech_video_done_with_warning(env):
    store, bus, settings, make = env
    fakes = Fakes(n_pieces=0)
    ds = _dataset(store, "v1")
    runner = make(fakes)
    _run(runner, ds)

    v = _video(store, ds, "v1")
    assert v["status"] == "done"
    assert v["warning"] == "<30 s clean audio"
    assert store.clips(ds, video_id="v1") == []
    assert "model_a" not in runner.gpu.used and "ecapa" not in runner.gpu.used  # nothing to load models for
    assert len(bus.of("video_done")) == 1


def test_short_kept_audio_warns(env):
    store, bus, settings, make = env
    fakes = Fakes(n_pieces=6)  # two 13 s clips
    ds = _dataset(store, "v1")
    _run(make(fakes), ds)
    assert _video(store, ds, "v1")["warning"] == "<30 s clean audio"


def test_running_reset_on_start(env):
    store, bus, settings, make = env
    fakes = Fakes()
    ds = _dataset(store, "v1")
    for s in STAGES[:7]:  # through punctuate_pieces
        store.mark_stage(ds, "v1", s, True)
    store.replace_pieces(ds, "v1", [
        {"idx": i, "range_idx": 0, "start_s": i * 4.5, "end_s": i * 4.5 + 4, "text_a": default_texts("v1", i)[0],
         "text_b": default_texts("v1", i)[1], "b_compression": 1.0,
         "text": default_texts("v1", i)[0] + ("।" if i % 3 == 2 else "")} for i in range(12)
    ])
    store.update_video(ds, "v1", status="running")  # backend died mid-job

    runner = make(fakes)
    runner.start()
    assert runner.wait_idle(IDLE_S)

    assert stages_for(bus, "v1") == ["group", "quality", "select", "punctuate"]
    assert _video(store, ds, "v1")["status"] == "done"
    assert len([c for c in store.clips(ds, video_id="v1") if c["status"] == "auto"]) == 4


def _english_a_repeat_b(vid, i):
    """Pieces 0-1: both models empty (broken pieces). Then model A English and model B a loop of one word:
    each piece passes on its own (B is fine), but every clip of them has both texts broken."""
    if i < 2:
        return "", ""
    return "hello this is english speech", "আচ্ছা আচ্ছা আচ্ছা" + (" end" if i == 11 else "")


def test_broken_pieces_and_bad_stt(env):
    store, bus, settings, make = env
    fakes = Fakes(texts=_english_a_repeat_b)
    ds = _dataset(store, "v1")
    _run(make(fakes), ds)

    pieces = store.pieces(ds, "v1")
    assert [p["broken"] for p in pieces] == [1, 1] + [0] * 10
    clips = store.clips(ds, video_id="v1")
    assert clips and all(c["start_s"] >= pieces[1]["end_s"] for c in clips)  # broken audio is in no clip
    assert all(c["status"] == "dropped" and c["drop_reason"] == "bad_stt" for c in clips)
    v = _video(store, ds, "v1")
    assert v["status"] == "done" and v["warning"] == "<30 s clean audio; poor STT"


def test_llm_down_flags_unsplit_and_retry_regroups(env):
    store, bus, settings, make = env
    fakes = Fakes()
    fakes.llm_up = False
    ds = _dataset(store, "v1")
    runner = make(fakes)
    _run(runner, ds)
    clips = store.clips(ds, video_id="v1")
    live = [c for c in clips if c["status"] != "dropped"]
    assert live and all(c["flag"] == "unsplit" and c["status"] == "needs_review" for c in live)
    assert all(c["no_punct"] == 1 and c["text_final"] == _kept(c) for c in live)

    fakes.llm_up = True
    fakes.order.clear()
    runner.retry_llm(ds)
    assert runner.wait_idle(IDLE_S)
    clips = store.clips(ds, video_id="v1")
    live = [c for c in clips if c["status"] != "dropped"]
    assert len(live) == 4
    assert all(c["flag"] is None and c["status"] == "auto" for c in live)
    assert all(c["no_punct"] == 0 and c["text_final"] == _kept(c) + "।" for c in live)
    assert all(p["no_punct"] == 0 for p in store.pieces(ds, "v1"))
    assert _video(store, ds, "v1")["status"] == "done"
    # the retry obeys the GPU tenancy too
    assert fakes.order[0] == ("release",) and fakes.order[-1] == ("llm_unload",)


def test_llm_retry_skips_regroup_when_a_clip_was_reviewed(env):
    store, bus, settings, make = env
    fakes = Fakes()
    fakes.llm_up = False
    ds = _dataset(store, "v1")
    runner = make(fakes)
    _run(runner, ds)
    first = store.clips(ds, video_id="v1")[0]
    store.update_clip(ds, first["id"], status="approved")
    before = store.clips(ds, video_id="v1")

    fakes.llm_up = True
    runner.retry_llm(ds)
    assert runner.wait_idle(IDLE_S)
    after = store.clips(ds, video_id="v1")
    assert [(c["id"], c["start_s"], c["end_s"], c["flag"]) for c in after] == \
        [(c["id"], c["start_s"], c["end_s"], c["flag"]) for c in before]
    assert after[0]["status"] == "approved"


def test_model_change_mid_stage_keeps_started_model(env):
    store, bus, settings, make = env
    fakes = Fakes()
    gpu = FakeGpu(fakes.order, options={"model_a": lambda: store.get_setting("model_a") or "fake-a"})
    ds = _dataset(store, "v1")
    orig = fakes.transcribe_a

    def transcribe_a(pieces, vocals_wav, model, on_progress):
        if _vid(vocals_wav) == "v1":  # Settings change while the stage runs
            store.set_setting("model_a", "new-a")
            gpu.unload("model_a")
        return orig(pieces, vocals_wav, model, on_progress)

    fakes.transcribe_a = transcribe_a
    runner = JobRunner(store, bus, gpu, llm=object(), settings=settings, stages=fakes.stage_fns())
    try:
        _run(runner, ds)
        assert _video(store, ds, "v1")["model_a"] == "fake-a"
        store.add_video(ds, "v2", "u2")
        runner.enqueue(ds)
        assert runner.wait_idle(IDLE_S)
        assert _video(store, ds, "v2")["model_a"] == "new-a"  # the next run uses the new model
    finally:
        runner.stop()


def test_retry_reruns_failed_stage(env):
    store, bus, settings, make = env
    fakes = Fakes(fail={("pieces", "v1"): RuntimeError("boom")})
    ds = _dataset(store, "v1")
    runner = make(fakes)
    _run(runner, ds)
    assert _video(store, ds, "v1")["status"] == "failed"

    fakes.fail = {}
    runner.retry(ds, "v1")
    assert runner.wait_idle(IDLE_S)
    v = _video(store, ds, "v1")
    assert v["status"] == "done" and v["error"] is None
    assert fakes.calls_for("v1").count("download") == 1  # download/cut/separate not rerun
    assert fakes.calls_for("v1").count("separate") == 1


def test_skip_video(env):
    store, bus, settings, make = env
    fakes = Fakes()
    ds = _dataset(store, "v1", "v2")
    runner = make(fakes)
    runner.skip(ds, "v1")
    _run(runner, ds)
    assert _video(store, ds, "v1")["status"] == "skipped"
    assert fakes.calls_for("v1") == []
    assert _video(store, ds, "v2")["status"] == "done"


def test_retry_llm_punctuates_no_punct_clips(env):
    store, bus, settings, make = env
    fakes = Fakes()
    fakes.llm_down_if = lambda text: text.count("আমি") > 1  # clip-level punctuation fails, piece-level works
    ds = _dataset(store, "v1")
    runner = make(fakes)
    _run(runner, ds)
    live = [c for c in store.clips(ds, video_id="v1") if c["status"] != "dropped"]
    assert all(c["flag"] is None and c["no_punct"] == 1 for c in live)

    fakes.llm_down_if = None
    runner.retry_llm(ds)
    assert runner.wait_idle(IDLE_S)
    live = [c for c in store.clips(ds, video_id="v1") if c["status"] != "dropped"]
    assert all(c["no_punct"] == 0 and c["text_final"] == _kept(c) + "।" for c in live)
    assert _video(store, ds, "v1")["status"] == "done"


def test_vocals_and_cut_path_layout(tmp_path):
    assert vocals_path(tmp_path, "ds1", "abc") == tmp_path / "videos" / "ds1" / "abc" / "separate" / "vocals.wav"
    assert cut_dir(tmp_path, "ds1", "abc") == tmp_path / "videos" / "ds1" / "abc" / "cut"


def test_review_edit_during_punctuate_survives(env):
    store, bus, settings, make = env
    fakes = Fakes()
    ds = _dataset(store, "v1")
    orig = fakes.punctuate
    acted = []

    def punctuate(text, llm):  # the reviewer acts after the clip-level punctuate took its snapshot
        if text.count("আমি") > 1 and not acted:
            acted.append(1)
            store.update_clip(ds, "v1_0002", text_final="edited")
            store.update_clip(ds, "v1_0003", text_final="human", status="approved")
        return orig(text, llm)

    fakes.punctuate = punctuate
    _run(make(fakes), ds)
    clips = {c["id"]: c for c in store.clips(ds, video_id="v1")}
    assert (clips["v1_0002"]["status"], clips["v1_0002"]["text_final"]) == ("auto", "edited")
    assert (clips["v1_0003"]["status"], clips["v1_0003"]["text_final"]) == ("approved", "human")
    assert clips["v1_0000"]["text_final"] == _kept(clips["v1_0000"]) + "।"
    assert _video(store, ds, "v1")["status"] == "done"


def test_retry_llm_keeps_review_edits_and_approvals(env):
    store, bus, settings, make = env
    fakes = Fakes()
    fakes.llm_down_if = lambda text: text.count("আমি") > 1
    ds = _dataset(store, "v1")
    runner = make(fakes)
    _run(runner, ds)
    c1 = store.get_clip(ds, "v1_0001")
    store.update_clip(ds, "v1_0000", text_final="fixed words")
    store.update_clip(ds, "v1_0001", text_final=_kept(c1), status="approved")

    fakes.llm_down_if = None
    runner.retry_llm(ds)
    assert runner.wait_idle(IDLE_S)
    clips = {c["id"]: c for c in store.clips(ds, video_id="v1")}
    assert clips["v1_0000"]["text_final"] == "fixed words"
    assert (clips["v1_0001"]["status"], clips["v1_0001"]["text_final"]) == ("approved", _kept(c1))
    assert clips["v1_0002"]["text_final"] == _kept(clips["v1_0002"]) + "।" and clips["v1_0002"]["no_punct"] == 0


def test_legacy_clip_punctuate_reads_text_a_when_no_pick(env):
    store, bus, settings, make = env
    fakes = Fakes()
    ds = _dataset(store, "v1")
    for s in STAGES[:-1]:
        store.mark_stage(ds, "v1", s, True)
    store.replace_clips(ds, "v1", [
        {"id": "v1_0000", "idx": 0, "start_s": 0, "end_s": 40, "text_a": "v1 old end", "text_b": "other",
         "status": "auto"},
        {"id": "v1_0001", "idx": 1, "start_s": 40, "end_s": 50, "text_a": "x", "text_source": "translated",
         "text_final": "bn", "status": "needs_review"},
    ])
    _run(make(fakes), ds)
    clips = {c["id"]: c for c in store.clips(ds, video_id="v1")}
    assert clips["v1_0000"]["text_final"] == "v1 old end।"
    assert clips["v1_0001"]["text_final"] == "bn"  # a translated legacy clip is left as it is


def test_default_transcribe_b_adapter(tmp_path):
    wav = tmp_path / "v.wav"
    sf.write(str(wav), np.zeros(16000 * 3, dtype="float32"), 16000)
    seen, progress = {}, []

    def model(audio16k, pieces):
        seen.update(n=len(audio16k), pieces=pieces)
        return [{"text": "t", "tokens": 1, "compression": 1.0, "retry_failed": False}]

    fn = StageFns.default().transcribe_b
    out = fn([{"idx": 0, "start_s": 0.0, "end_s": 2.0}], wav, model, lambda d, t: progress.append((d, t)))
    assert out == [{"text": "t", "tokens": 1, "compression": 1.0, "retry_failed": False}]
    assert seen["n"] == 48000 and seen["pieces"] == [{"idx": 0, "start_s": 0.0, "end_s": 2.0}]
    assert progress == [(1, 1)]


def test_default_llm_unload_calls_client():
    calls = []

    class Llm:
        def unload(self):
            calls.append(1)

    StageFns.default().llm_unload(Llm())
    StageFns.default().llm_unload(object())  # a client without unload is fine
    assert calls == [1]


class FlakyStore(Store):
    """Raises sqlite3.OperationalError once from a chosen store call made by a runner thread."""

    def __init__(self, path):
        super().__init__(path)
        self.fail_failed_update = 0   # update_video(status="failed") calls left to fail
        self.fail_stage_done = 0      # stage_done calls (scheduling reads) from ytds-* threads left to fail

    def update_video(self, dataset_id, video_id, **fields):
        if fields.get("status") == "failed" and self.fail_failed_update:
            self.fail_failed_update -= 1
            raise sqlite3.OperationalError("database or disk is full")
        return super().update_video(dataset_id, video_id, **fields)

    def stage_done(self, dataset_id, video_id, stage):
        if self.fail_stage_done and threading.current_thread().name.startswith("ytds-"):
            self.fail_stage_done -= 1
            raise sqlite3.OperationalError("database is locked")
        return super().stage_done(dataset_id, video_id, stage)


def _flaky_runner(tmp_path, fakes):
    store = FlakyStore(tmp_path / "flaky.sqlite")
    bus = FakeBus()
    r = JobRunner(store, bus, FakeGpu(fakes.order), llm=object(), settings=Settings(data_dir=tmp_path / "var"),
                  stages=fakes.stage_fns())
    return store, bus, r


@pytest.mark.parametrize("stage", ["separate", "pieces"])  # GPU thread, pool thread
def test_store_error_while_failing_a_video_keeps_worker_alive(tmp_path, stage):
    fakes = Fakes(fail={(stage, "v1"): RuntimeError("stage boom")})
    store, bus, runner = _flaky_runner(tmp_path, fakes)
    store.fail_failed_update = 1
    ds = _dataset(store, "v1", "v2", "v3")
    try:
        _run(runner, ds)
        assert all(t.is_alive() for t in runner._threads)
        v1 = _video(store, ds, "v1")
        assert v1["status"] == "failed" and v1["error"] == "stage boom"
        assert all(_video(store, ds, v)["status"] == "done" for v in ("v2", "v3"))
        assert bus.of("log")
    finally:
        runner.stop()


def test_store_error_in_scheduling_keeps_workers_alive(tmp_path):
    fakes = Fakes()
    store, bus, runner = _flaky_runner(tmp_path, fakes)
    ds = _dataset(store, "v1", "v2")
    orig = fakes.download

    def download(video_id, url, out_dir, on_progress):
        store.fail_stage_done = 3  # the next scheduling reads in runner threads fail
        return orig(video_id, url, out_dir, on_progress)

    runner.stages.download = download
    try:
        _run(runner, ds)
        assert all(t.is_alive() for t in runner._threads)
        assert all(_video(store, ds, v)["status"] == "done" for v in ("v1", "v2"))
        assert bus.of("log")
    finally:
        runner.stop()


def test_busy_until_running_stage_ends_even_after_skip(env):
    store, bus, settings, make = env
    fakes = Fakes()
    ds = _dataset(store, "v1")
    entered, release = threading.Event(), threading.Event()
    orig = fakes.pieces

    def pieces(vocals_wav, cmap, th):
        entered.set()
        assert release.wait(IDLE_S)
        return orig(vocals_wav, cmap, th)

    fakes.pieces = pieces
    runner = make(fakes)
    assert runner.busy(ds, "v1") is False
    runner.start()
    runner.enqueue(ds)
    assert entered.wait(IDLE_S)
    runner.skip(ds, "v1")
    assert runner.busy(ds, "v1") is True  # Skip does not stop the running stage
    release.set()
    assert runner.wait_idle(IDLE_S)
    assert runner.busy(ds, "v1") is False
    assert _video(store, ds, "v1")["status"] == "skipped"


def test_skip_cancels_running_stage(env):
    store, bus, settings, make = env
    fakes = Fakes()
    ds = _dataset(store, "v1", "v2")
    entered = threading.Event()
    orig = fakes.separate

    def separate(audio_wav, out_wav, model, segment_s, on_progress):
        if _vid(out_wav) != "v1":
            return orig(audio_wav, out_wav, model, segment_s, on_progress)
        entered.set()
        for i in range(10_000):  # a long stage that reports progress; only a cancel ends it early
            on_progress(i, 10_000)
            time.sleep(0.01)
        raise AssertionError("stage was not cancelled")

    fakes.separate = separate
    runner = make(fakes)
    runner.start()
    try:
        runner.enqueue(ds)
        assert entered.wait(IDLE_S)
        runner.skip(ds, "v1")
        assert runner.wait_idle(IDLE_S)
        v1 = _video(store, ds, "v1")
        assert v1["status"] == "skipped" and v1["error"] is None
        assert not store.stage_done(ds, "v1", "separate")
        assert "pieces" not in fakes.calls_for("v1")
        assert runner.busy(ds, "v1") is False
        assert _video(store, ds, "v2")["status"] == "done"
        assert not bus.of("stage_failed")
    finally:
        runner.stop()


def test_retry_after_cancel_runs_the_stage_again(env):
    store, bus, settings, make = env
    fakes = Fakes()
    ds = _dataset(store, "v1")
    runner = make(fakes)
    runner.skip(ds, "v1")  # not in flight: nothing to cancel, and no cancel left behind
    runner.retry(ds, "v1")
    _run(runner, ds)
    assert _video(store, ds, "v1")["status"] == "done"
