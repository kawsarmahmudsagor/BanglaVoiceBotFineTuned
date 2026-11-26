import sqlite3
import threading
import time
from pathlib import Path

import pytest

from ytds.config import Settings
from ytds.pipeline import STAGES
from ytds.runner import JobRunner, StageFns, vocals_path
from ytds.store import Store

IDLE_S = 10


class FakeBus:
    def __init__(self):
        self.events, self._lock = [], threading.Lock()

    def publish(self, event):
        with self._lock:
            self.events.append(dict(event))

    def of(self, type_):
        with self._lock:
            return [e for e in self.events if e["type"] == type_]


class FakeGpu:
    def __init__(self):
        self.used = []

    def use(self, name):
        self.used.append(name)
        return name


def _vid(path) -> str:
    return Path(path).parent.parent.name  # <dataset>/<video>/<stage>/<file>


class Fakes:
    """Stage fakes with the real stage signatures; `log` records (stage, video_id) in call order."""

    def __init__(self, n_clips=4, clip_s=10.0, fail=None, routes=None):
        self.log, self._lock = [], threading.Lock()
        self.n_clips, self.clip_s = n_clips, clip_s
        self.fail = fail or {}  # (stage, video_id) -> exception
        self.routes = routes  # per-clip status set by transcribe, default all "auto"
        self.llm_up = True

    def _rec(self, stage, vid):
        with self._lock:
            self.log.append((stage, vid))
        exc = self.fail.get((stage, vid))
        if exc:
            raise exc

    def download(self, video_id, url, out_dir, on_progress):
        self._rec("download", video_id)
        on_progress(50, 100)
        on_progress(100, 100)
        return {"title": f"title {video_id}", "duration_s": 60.0}

    def separate(self, audio_wav, out_wav, model, segment_s, on_progress):
        assert model == "demucs"
        self._rec("separate", _vid(out_wav))
        on_progress(1, 1)

    def segment(self, vocals_wav, th):
        self._rec("segment", _vid(vocals_wav))
        return [{"idx": i, "start_s": i * 20.0, "end_s": i * 20.0 + self.clip_s} for i in range(self.n_clips)]

    def quality(self, vocals_wav, clips, th, scorer=None, embedder=None):
        assert embedder == "ecapa"
        self._rec("quality", _vid(vocals_wav))
        for c in clips:
            c.update(dnsmos=3.5, speaker_sim=0.9, drop_reason=None)
        return clips

    def transcribe(self, clips, vocals_wav, gpu, on_progress, th=None):
        assert th is not None
        vid = _vid(vocals_wav)
        self._rec("transcribe", vid)
        for i, c in enumerate(clips):
            status = self.routes[i] if self.routes else "auto"
            c.update(text_a=f"{vid} text {i}", text_b="b", cer=0.0, text_source="stt", status=status)
        on_progress(len(clips), len(clips))

    def fallback(self, clips, vocals_wav, gpu, llm, on_progress):
        self._rec("fallback", _vid(vocals_wav))
        for c in clips:
            if c["status"] == "pending_llm" and self.llm_up:
                c.update(text_en="en", text_translated="bn", text_final="bn",
                         text_source="translated", status="needs_review")

    def punctuate(self, text, llm):
        self._rec("punctuate", text.split()[0])
        if not self.llm_up:
            return text, True
        return text + "।", False

    def stage_fns(self):
        return StageFns(download=self.download, separate=self.separate, segment=self.segment,
                        quality=self.quality, transcribe=self.transcribe, fallback=self.fallback,
                        punctuate=self.punctuate)

    def stages_for(self, vid):
        with self._lock:
            seq = [s for s, v in self.log if v == vid]
        out = []
        for s in seq:  # punctuate is called once per clip
            if not out or out[-1] != s:
                out.append(s)
        return out


@pytest.fixture
def env(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    bus = FakeBus()
    settings = Settings(data_dir=tmp_path / "var")
    runners = []

    def make(fakes):
        r = JobRunner(store, bus, FakeGpu(), llm=object(), settings=settings, stages=fakes.stage_fns())
        runners.append(r)
        return r

    yield store, bus, settings, make
    for r in runners:
        r.stop()


def _dataset(store, *vids):
    ds = store.create_dataset("d", "spk")
    for v in vids:
        store.add_video(ds, v, f"https://youtu.be/{v}")
    return ds


def _video(store, ds, vid):
    return next(v for v in store.list_videos(ds) if v["id"] == vid)


def _run(runner, ds):
    runner.start()
    runner.enqueue(ds)
    assert runner.wait_idle(IDLE_S)


def test_runs_all_stages_in_order(env):
    store, bus, settings, make = env
    fakes = Fakes()
    ds = _dataset(store, "v1")
    _run(make(fakes), ds)

    assert fakes.stages_for("v1") == list(STAGES)
    v = _video(store, ds, "v1")
    assert v["status"] == "done" and v["warning"] is None and v["error"] is None
    assert v["title"] == "title v1"
    assert all(store.stage_done(ds, "v1", s) for s in STAGES)
    clips = store.clips(ds, video_id="v1")
    assert [c["id"] for c in clips] == [f"v1_{i:04d}" for i in range(4)]
    assert all(c["text_final"] == c["text_a"] + "।" and c["no_punct"] == 0 for c in clips)
    assert [e["stage"] for e in bus.of("stage_done")] == list(STAGES)
    assert len(bus.of("video_done")) == 1
    progress = bus.of("progress")
    assert {"dataset_id": ds, "video_id": "v1", "stage": "download", "done": 50, "total": 100}.items() \
        <= progress[0].items()
    assert "eta_s" in progress[0]


def test_resume_skips_done_stages(env):
    store, bus, settings, make = env
    fakes = Fakes()
    ds = _dataset(store, "v1")
    store.mark_stage(ds, "v1", "download", True)
    store.mark_stage(ds, "v1", "separate", True)
    _run(make(fakes), ds)

    assert fakes.stages_for("v1") == list(STAGES[2:])
    assert _video(store, ds, "v1")["status"] == "done"


def test_failure_isolated_to_video(env):
    store, bus, settings, make = env
    fakes = Fakes(fail={("segment", "v1"): RuntimeError("vad exploded")})
    ds = _dataset(store, "v1", "v2")
    _run(make(fakes), ds)

    v1, v2 = _video(store, ds, "v1"), _video(store, ds, "v2")
    assert v1["status"] == "failed" and v1["error"] == "vad exploded"
    assert v2["status"] == "done"
    assert fakes.stages_for("v1") == ["download", "separate", "segment"]
    assert not store.stage_done(ds, "v1", "segment")
    failed = bus.of("stage_failed")
    assert len(failed) == 1
    assert failed[0]["video_id"] == "v1" and failed[0]["stage"] == "segment"
    assert failed[0]["error"] == "vad exploded"


def test_gpu_stage_major_across_videos(env):
    store, bus, settings, make = env
    fakes = Fakes()
    ds = _dataset(store, "v1", "v2")
    for v in ("v1", "v2"):
        store.mark_stage(ds, v, "download", True)
    _run(make(fakes), ds)

    seq = [s for s, _ in fakes.log]
    last_separate = max(i for i, s in enumerate(seq) if s == "separate")
    first_transcribe = min(i for i, s in enumerate(seq) if s == "transcribe")
    assert seq.count("separate") == 2
    assert last_separate < first_transcribe
    assert all(_video(store, ds, v)["status"] == "done" for v in ("v1", "v2"))


def test_no_speech_video_done_with_warning(env):
    store, bus, settings, make = env
    fakes = Fakes(n_clips=0)
    ds = _dataset(store, "v1")
    _run(make(fakes), ds)

    v = _video(store, ds, "v1")
    assert v["status"] == "done"
    assert v["warning"] == "<30 s clean audio"
    assert store.clips(ds, video_id="v1") == []
    assert len(bus.of("video_done")) == 1


def test_short_kept_audio_warns(env):
    store, bus, settings, make = env
    fakes = Fakes(n_clips=2, clip_s=10.0)  # 20 s kept
    ds = _dataset(store, "v1")
    _run(make(fakes), ds)
    assert _video(store, ds, "v1")["warning"] == "<30 s clean audio"


def test_running_reset_on_start(env):
    store, bus, settings, make = env
    fakes = Fakes()
    ds = _dataset(store, "v1")
    for s in STAGES[:5]:
        store.mark_stage(ds, "v1", s, True)
    store.replace_clips(ds, "v1", [
        {"id": f"v1_{i:04d}", "idx": i, "start_s": i * 20.0, "end_s": i * 20.0 + 10, "text_a": f"v1 t{i}",
         "status": "auto"} for i in range(4)
    ])
    store.update_video(ds, "v1", status="running")  # backend died mid-job

    runner = make(fakes)
    runner.start()  # no enqueue: start alone resumes
    assert runner.wait_idle(IDLE_S)

    assert fakes.stages_for("v1") == ["fallback", "punctuate"]
    assert _video(store, ds, "v1")["status"] == "done"


def test_poor_stt_warning(env):
    store, bus, settings, make = env
    fakes = Fakes(routes=["pending_llm", "pending_llm", "pending_llm", "auto"])
    ds = _dataset(store, "v1")
    _run(make(fakes), ds)

    v = _video(store, ds, "v1")
    assert v["status"] == "done" and v["warning"] == "poor STT"
    statuses = [c["status"] for c in store.clips(ds, video_id="v1")]
    assert statuses == ["needs_review"] * 3 + ["auto"]


def test_both_warnings_joined(env):
    store, bus, settings, make = env
    fakes = Fakes(n_clips=2, routes=["pending_llm", "pending_llm"])
    ds = _dataset(store, "v1")
    _run(make(fakes), ds)
    assert _video(store, ds, "v1")["warning"] == "<30 s clean audio; poor STT"


def test_retry_reruns_failed_stage(env):
    store, bus, settings, make = env
    fakes = Fakes(fail={("segment", "v1"): RuntimeError("boom")})
    ds = _dataset(store, "v1")
    runner = make(fakes)
    _run(runner, ds)
    assert _video(store, ds, "v1")["status"] == "failed"

    fakes.fail = {}
    runner.retry(ds, "v1")
    assert runner.wait_idle(IDLE_S)
    v = _video(store, ds, "v1")
    assert v["status"] == "done" and v["error"] is None
    assert fakes.stages_for("v1") == list(STAGES)  # download/separate not rerun


def test_skip_video(env):
    store, bus, settings, make = env
    fakes = Fakes()
    ds = _dataset(store, "v1", "v2")
    runner = make(fakes)
    runner.skip(ds, "v1")
    _run(runner, ds)
    assert _video(store, ds, "v1")["status"] == "skipped"
    assert fakes.stages_for("v1") == []
    assert _video(store, ds, "v2")["status"] == "done"


def test_retry_llm(env):
    store, bus, settings, make = env
    fakes = Fakes(routes=["pending_llm", "auto", "auto", "auto"])
    fakes.llm_up = False
    ds = _dataset(store, "v1")
    runner = make(fakes)
    _run(runner, ds)
    clips = store.clips(ds, video_id="v1")
    assert clips[0]["status"] == "pending_llm"
    assert all(c["no_punct"] == 1 for c in clips[1:])

    fakes.llm_up = True
    runner.retry_llm(ds)
    assert runner.wait_idle(IDLE_S)
    clips = store.clips(ds, video_id="v1")
    assert clips[0]["status"] == "needs_review" and clips[0]["text_final"] == "bn"
    assert all(c["no_punct"] == 0 and c["text_final"] == c["text_a"] + "।" for c in clips[1:])
    assert _video(store, ds, "v1")["status"] == "done"


def test_vocals_path_layout(tmp_path):
    assert vocals_path(tmp_path, "ds1", "abc") == tmp_path / "videos" / "ds1" / "abc" / "separate" / "vocals.wav"


def test_review_edit_during_punctuate_survives(env):
    store, bus, settings, make = env
    fakes = Fakes(routes=["flagged"] * 4)
    ds = _dataset(store, "v1")
    orig = fakes.punctuate

    def punctuate(text, llm):  # the reviewer acts after punctuate took its snapshot
        if not any(s == "punctuate" for s, _ in fakes.log):
            store.update_clip(ds, "v1_0002", text_final="edited")
            store.update_clip(ds, "v1_0003", text_final="human", status="approved")
        return orig(text, llm)

    fakes.punctuate = punctuate
    _run(make(fakes), ds)
    clips = {c["id"]: c for c in store.clips(ds, video_id="v1")}
    assert (clips["v1_0002"]["status"], clips["v1_0002"]["text_final"]) == ("flagged", "edited")
    assert (clips["v1_0003"]["status"], clips["v1_0003"]["text_final"]) == ("approved", "human")
    assert clips["v1_0000"]["text_final"] == clips["v1_0000"]["text_a"] + "।"
    assert _video(store, ds, "v1")["status"] == "done"


def test_review_edit_during_fallback_survives(env):
    store, bus, settings, make = env
    fakes = Fakes(routes=["pending_llm", "pending_llm", "pending_llm", "auto"])
    ds = _dataset(store, "v1")
    orig = fakes.fallback

    def fallback(clips, vocals_wav, gpu, llm, on_progress):  # reviewer acts after the snapshot
        store.update_clip(ds, "v1_0000", text_final="human")
        store.update_clip(ds, "v1_0001", text_final="ok", status="approved")
        return orig(clips, vocals_wav, gpu, llm, on_progress)

    fakes.fallback = fallback
    _run(make(fakes), ds)
    clips = {c["id"]: c for c in store.clips(ds, video_id="v1")}
    assert (clips["v1_0000"]["status"], clips["v1_0000"]["text_final"]) == ("pending_llm", "human")
    assert (clips["v1_0001"]["status"], clips["v1_0001"]["text_final"]) == ("approved", "ok")
    assert (clips["v1_0002"]["status"], clips["v1_0002"]["text_final"]) == ("needs_review", "bn")


def test_retry_llm_keeps_review_edits_and_approvals(env):
    store, bus, settings, make = env
    fakes = Fakes(routes=["pending_llm", "flagged", "flagged", "auto"])
    fakes.llm_up = False
    ds = _dataset(store, "v1")
    runner = make(fakes)
    _run(runner, ds)
    store.update_clip(ds, "v1_0000", text_final="typed on pending")
    store.update_clip(ds, "v1_0001", text_final="fixed words")
    store.update_clip(ds, "v1_0002", text_final="v1 text 2", status="approved")

    fakes.llm_up = True
    runner.retry_llm(ds)
    assert runner.wait_idle(IDLE_S)
    clips = {c["id"]: c for c in store.clips(ds, video_id="v1")}
    assert (clips["v1_0000"]["status"], clips["v1_0000"]["text_final"]) == ("pending_llm", "typed on pending")
    assert (clips["v1_0001"]["status"], clips["v1_0001"]["text_final"]) == ("flagged", "fixed words")
    assert (clips["v1_0002"]["status"], clips["v1_0002"]["text_final"]) == ("approved", "v1 text 2")
    assert clips["v1_0003"]["text_final"] == "v1 text 3।" and clips["v1_0003"]["no_punct"] == 0


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
    r = JobRunner(store, bus, FakeGpu(), llm=object(), settings=Settings(data_dir=tmp_path / "var"),
                  stages=fakes.stage_fns())
    return store, bus, r


@pytest.mark.parametrize("stage", ["separate", "segment"])  # GPU thread, pool thread
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
    orig = fakes.segment

    def segment(vocals_wav, th):
        entered.set()
        assert release.wait(IDLE_S)
        return orig(vocals_wav, th)

    fakes.segment = segment
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
        assert "segment" not in [s for s, vid in fakes.log if vid == "v1"]
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
