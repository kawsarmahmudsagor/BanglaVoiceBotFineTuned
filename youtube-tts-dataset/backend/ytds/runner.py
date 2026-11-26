"""JobRunner: per-video stage graph with resume, a 2-thread CPU/network pool and one GPU thread.

Per-video files live under `<data_dir>/videos/<dataset_id>/<video_id>/`:
  download/audio.wav     48 kHz mono (download stage)
  separate/vocals.wav    44.1 kHz mono vocals stem (separate stage)
Clip ids are `<video_id>_<idx:04d>`.

Stage placement (ruling R3): the pool runs download, segment, punctuate and the final bookkeeping; the GPU
thread runs separate, quality (gpu.use("ecapa") as the embedder), transcribe and fallback, stage-major: it
takes the first GPU stage in STAGES order that has ready videos and runs it for all of them before moving on.
A stage is marked done only after its outputs are persisted, so a restart reruns unfinished stages.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .gpu import run_with_oom_retry
from .pipeline import STAGES

GPU_STAGES = ("separate", "quality", "transcribe", "fallback")
FINALIZE = "finalize"
PUNCT_RETRY = "punctuate_retry"
POOL_WORKERS = 2
MIN_CLEAN_S = 30.0
WARN_SHORT = "<30 s clean audio"
WARN_POOR_STT = "poor STT"

_ACTIVE = ("queued", "running")
_QUALITY_FIELDS = ("dnsmos", "speaker_sim", "drop_reason", "status")
_TRANSCRIBE_FIELDS = ("text_a", "text_b", "cer", "text_source", "status")
_FALLBACK_FIELDS = ("text_en", "text_translated", "text_final", "text_source", "status")


def video_dir(data_dir: Path, dataset_id: str, video_id: str) -> Path:
    return Path(data_dir) / "videos" / dataset_id / video_id


def audio_path(data_dir: Path, dataset_id: str, video_id: str) -> Path:
    return video_dir(data_dir, dataset_id, video_id) / "download" / "audio.wav"


def vocals_path(data_dir: Path, dataset_id: str, video_id: str) -> Path:
    return video_dir(data_dir, dataset_id, video_id) / "separate" / "vocals.wav"


@dataclass
class StageFns:
    """The stage callables, with the real stage modules' signatures (injectable for tests)."""

    download: Callable[..., dict]       # (video_id, url, out_dir, on_progress) -> metadata
    separate: Callable[..., None]       # (audio_wav, out_wav, model, segment_s, on_progress)
    segment: Callable[..., list]        # (vocals_wav, th) -> [{idx, start_s, end_s}]
    quality: Callable[..., list]        # (vocals_wav, clips, th, embedder=) -> clips
    transcribe: Callable[..., None]     # (clips, vocals_wav, gpu, on_progress, th=)
    fallback: Callable[..., None]       # (clips, vocals_wav, gpu, llm, on_progress)
    punctuate: Callable[..., tuple]     # (text, llm) -> (text, no_punct)

    @classmethod
    def default(cls) -> StageFns:
        from .pipeline.download import download
        from .pipeline.fallback import run_fallback
        from .pipeline.punct import punctuate
        from .pipeline.quality import quality
        from .pipeline.segment import segment
        from .pipeline.separate import separate
        from .pipeline.transcribe import transcribe_clips

        return cls(download=download, separate=separate, segment=segment, quality=quality,
                   transcribe=transcribe_clips, fallback=run_fallback, punctuate=punctuate)


class JobRunner:
    def __init__(self, store, bus, gpu, llm, settings, stages: StageFns | None = None):
        self.store, self.bus, self.gpu, self.llm, self.settings = store, bus, gpu, llm, settings
        self.stages = stages or StageFns.default()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._tasks: queue.Queue = queue.Queue()
        self._active: set[str] = set()          # dataset ids being scheduled
        self._inflight: set[tuple] = set()      # (dataset_id, video_id) or (dataset_id, video_id, PUNCT_RETRY)
        self._llm_retry: list[str] = []         # dataset ids waiting for the GPU thread
        self._llm_busy = 0
        self._t0: dict[tuple, float] = {}
        self._threads: list[threading.Thread] = []

    # --- public API
    def start(self) -> None:
        for ds in self.store.list_datasets():
            for v in self.store.list_videos(ds["id"]):
                if v["status"] == "running":
                    self.store.update_video(ds["id"], v["id"], status="queued")
                if v["status"] in _ACTIVE:
                    with self._lock:
                        self._active.add(ds["id"])
        self._stop.clear()
        self._threads = [threading.Thread(target=self._gpu_loop, name="ytds-gpu", daemon=True)]
        self._threads += [threading.Thread(target=self._pool_loop, name=f"ytds-pool-{i}", daemon=True)
                          for i in range(POOL_WORKERS)]
        for t in self._threads:
            t.start()
        self._kick()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        for t in self._threads:
            t.join(timeout)
        self._threads = []

    def enqueue(self, dataset_id: str) -> None:
        with self._lock:
            self._active.add(dataset_id)
        self._kick()

    def retry(self, dataset_id: str, video_id: str) -> None:
        step = self._next_step(dataset_id, video_id)
        if step != FINALIZE:
            self.store.mark_stage(dataset_id, video_id, step, False)
        self.store.update_video(dataset_id, video_id, status="queued", error=None)
        self.enqueue(dataset_id)

    def skip(self, dataset_id: str, video_id: str) -> None:
        self.store.update_video(dataset_id, video_id, status="skipped")

    def retry_llm(self, dataset_id: str) -> None:
        """Rerun fallback for pending_llm clips and punctuate for no_punct auto/flagged clips, in videos
        whose stage already ran."""
        with self._cv:
            self._llm_retry.append(dataset_id)
            self._cv.notify_all()

    def wait_idle(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._idle():
                return True
            time.sleep(0.02)
        return self._idle()

    # --- scheduling
    def _idle(self) -> bool:
        with self._lock:
            if self._inflight or self._llm_retry or self._llm_busy:
                return False
            active = list(self._active)
        return not any(v["status"] in _ACTIVE for ds in active for v in self.store.list_videos(ds))

    def _next_step(self, ds: str, vid: str) -> str:
        for s in STAGES:
            if not self.store.stage_done(ds, vid, s):
                return s
        return FINALIZE

    def _ready(self) -> list[tuple[str, str, str]]:
        """(dataset_id, video_id, step) for active videos not in flight. Caller holds the lock."""
        out = []
        for ds in sorted(self._active):
            for v in self.store.list_videos(ds):
                if v["status"] in _ACTIVE and (ds, v["id"]) not in self._inflight:
                    out.append((ds, v["id"], self._next_step(ds, v["id"])))
        return out

    def _kick(self) -> None:
        with self._cv:
            for ds, vid, step in self._ready():
                if step not in GPU_STAGES:
                    self._inflight.add((ds, vid))
                    self._tasks.put((ds, vid, step))
            self._cv.notify_all()

    def _pool_loop(self) -> None:
        while not self._stop.is_set():
            try:
                ds, vid, step = self._tasks.get(timeout=0.2)
            except queue.Empty:
                continue
            if step == PUNCT_RETRY:
                self._run_punct_retry(ds, vid)
            elif step == FINALIZE:
                self._run_step(ds, vid, step, self._finalize)
            else:
                self._run_step(ds, vid, step, getattr(self, f"_stage_{step}"))

    def _gpu_loop(self) -> None:
        while not self._stop.is_set():
            with self._cv:
                retry_ds = self._llm_retry.pop(0) if self._llm_retry else None
                if retry_ds is not None:
                    self._llm_busy += 1
                batch = None if retry_ds is not None else self._gpu_batch()
                if retry_ds is None and batch is None:
                    self._cv.wait(0.5)
                    continue
            if batch is None:
                try:
                    self._run_llm_retry(retry_ds)
                finally:
                    with self._lock:
                        self._llm_busy -= 1
                continue
            stage, vids = batch
            for i, (ds, vid) in enumerate(vids):
                if self._stop.is_set():
                    with self._lock:
                        self._inflight.difference_update(vids[i:])
                    break
                self._run_step(ds, vid, stage, getattr(self, f"_stage_{stage}"))

    def _gpu_batch(self) -> tuple[str, list[tuple[str, str]]] | None:
        """First GPU stage in STAGES order with ready videos, and all of them. Caller holds the lock."""
        ready = [r for r in self._ready() if r[2] in GPU_STAGES]
        for stage in GPU_STAGES:
            vids = [(ds, vid) for ds, vid, step in ready if step == stage]
            if vids:
                self._inflight.update(vids)
                return stage, vids
        return None

    def _run_step(self, ds: str, vid: str, step: str, fn: Callable[[str, str], None]) -> None:
        try:
            if self._status(ds, vid) not in _ACTIVE:
                return  # skipped while waiting
            if step != FINALIZE:
                self.store.update_video(ds, vid, status="running")
                self._t0[(ds, vid, step)] = time.monotonic()
            fn(ds, vid)
            if step != FINALIZE:
                self.store.mark_stage(ds, vid, step, True)
                self._publish(type="stage_done", dataset_id=ds, video_id=vid, stage=step)
        except Exception as e:
            if self._status(ds, vid) != "skipped":
                self.store.update_video(ds, vid, status="failed", error=str(e))
            self._publish(type="stage_failed", dataset_id=ds, video_id=vid, stage=step, error=str(e))
        finally:
            self._t0.pop((ds, vid, step), None)
            with self._lock:
                self._inflight.discard((ds, vid))
            self._kick()

    # --- helpers
    def _status(self, ds: str, vid: str) -> str | None:
        for v in self.store.list_videos(ds):
            if v["id"] == vid:
                return v["status"]
        return None

    def _publish(self, **event: Any) -> None:
        try:
            self.bus.publish(event)
        except Exception:
            pass  # a closed event loop must not fail a stage

    def _progress(self, ds: str, vid: str, stage: str) -> Callable[[int, int], None]:
        def cb(done: int, total: int) -> None:
            eta = None
            t0 = self._t0.get((ds, vid, stage))
            if t0 is not None and done > 0 and total > done:
                eta = (time.monotonic() - t0) / done * (total - done)
            self._publish(type="progress", dataset_id=ds, video_id=vid, stage=stage,
                          done=done, total=total, eta_s=eta)
        return cb

    def _paths(self, ds: str, vid: str) -> tuple[Path, Path]:
        return audio_path(self.settings.data_dir, ds, vid), vocals_path(self.settings.data_dir, ds, vid)

    def _save(self, ds: str, clips: list[dict], fields: tuple[str, ...]) -> None:
        for c in clips:
            self.store.update_clip(ds, c["id"], **{k: c.get(k) for k in fields})

    # --- stages
    def _stage_download(self, ds: str, vid: str) -> None:
        url = next(v["url"] for v in self.store.list_videos(ds) if v["id"] == vid)
        audio, _ = self._paths(ds, vid)
        meta = self.stages.download(vid, url, audio.parent, self._progress(ds, vid, "download"))
        self.store.update_video(ds, vid, **meta)

    def _stage_separate(self, ds: str, vid: str) -> None:
        audio, vocals = self._paths(ds, vid)
        vocals.parent.mkdir(parents=True, exist_ok=True)
        cb = self._progress(ds, vid, "separate")
        run_with_oom_retry(
            lambda seg: self.stages.separate(audio, vocals, self.gpu.use("demucs"), seg, cb),
            self.store.get_thresholds().demucs_segment_s,
        )

    def _stage_segment(self, ds: str, vid: str) -> None:
        _, vocals = self._paths(ds, vid)
        clips = self.stages.segment(vocals, self.store.get_thresholds())
        self.store.replace_clips(ds, vid, [{**c, "id": f"{vid}_{c['idx']:04d}"} for c in clips])

    def _stage_quality(self, ds: str, vid: str) -> None:
        _, vocals = self._paths(ds, vid)
        clips = self.store.clips(ds, video_id=vid)
        if not clips:
            return  # no speech: do not load ECAPA for nothing
        clips = self.stages.quality(vocals, clips, self.store.get_thresholds(), embedder=self.gpu.use("ecapa"))
        self._save(ds, clips, _QUALITY_FIELDS)

    def _stage_transcribe(self, ds: str, vid: str) -> None:
        _, vocals = self._paths(ds, vid)
        clips = [c for c in self.store.clips(ds, video_id=vid) if c["status"] != "dropped"]
        self.stages.transcribe(clips, vocals, self.gpu, self._progress(ds, vid, "transcribe"),
                               th=self.store.get_thresholds())
        self._save(ds, clips, _TRANSCRIBE_FIELDS)

    def _stage_fallback(self, ds: str, vid: str) -> None:
        _, vocals = self._paths(ds, vid)
        clips = self.store.clips(ds, video_id=vid, statuses=["pending_llm"])
        self.stages.fallback(clips, vocals, self.gpu, self.llm, self._progress(ds, vid, "fallback"))
        self._save(ds, clips, _FALLBACK_FIELDS)

    def _stage_punctuate(self, ds: str, vid: str, only_no_punct: bool = False) -> None:
        clips = self.store.clips(ds, video_id=vid, statuses=["auto", "flagged"])
        if only_no_punct:
            clips = [c for c in clips if c["no_punct"]]
        cb = self._progress(ds, vid, "punctuate")
        for i, c in enumerate(clips, 1):
            text, no_punct = self.stages.punctuate(c["text_a"] or "", self.llm)
            self.store.update_clip(ds, c["id"], text_final=text, no_punct=int(no_punct))
            cb(i, len(clips))

    def _finalize(self, ds: str, vid: str) -> None:
        live = [c for c in self.store.clips(ds, video_id=vid) if c["status"] != "dropped"]
        kept_s = sum(c["end_s"] - c["start_s"] for c in live)
        fell_back = [c for c in live if c["status"] == "pending_llm" or c["text_source"] == "translated"]
        warnings = []
        if kept_s < MIN_CLEAN_S:
            warnings.append(WARN_SHORT)
        if live and len(fell_back) > len(live) / 2:
            warnings.append(WARN_POOR_STT)
        warning = "; ".join(warnings) or None
        self.store.update_video(ds, vid, status="done", warning=warning)
        self._publish(type="video_done", dataset_id=ds, video_id=vid, message=warning)

    # --- LLM retry
    def _run_llm_retry(self, ds: str) -> None:
        for v in self.store.list_videos(ds):
            vid = v["id"]
            if self.store.stage_done(ds, vid, "fallback"):
                try:
                    self._stage_fallback(ds, vid)
                except Exception as e:
                    self._publish(type="log", dataset_id=ds, video_id=vid, stage="fallback",
                                  message=f"LLM retry failed: {e}")
            if self.store.stage_done(ds, vid, "punctuate"):
                with self._lock:
                    self._inflight.add((ds, vid, PUNCT_RETRY))
                self._tasks.put((ds, vid, PUNCT_RETRY))

    def _run_punct_retry(self, ds: str, vid: str) -> None:
        try:
            self._stage_punctuate(ds, vid, only_no_punct=True)
        except Exception as e:
            self._publish(type="log", dataset_id=ds, video_id=vid, stage="punctuate",
                          message=f"LLM retry failed: {e}")
        finally:
            with self._lock:
                self._inflight.discard((ds, vid, PUNCT_RETRY))
