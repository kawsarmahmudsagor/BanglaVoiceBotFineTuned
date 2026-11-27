"""JobRunner: per-video stage graph with resume, a 2-thread CPU/network pool and one GPU thread.

Per-video files live under `<data_dir>/videos/<dataset_id>/<video_id>/`:
  download/audio.wav     48 kHz mono (download stage)
  cut/audio.wav          the kept ranges joined with 2 s gaps, and cut/map.json, its time map (cut stage)
  separate/vocals.wav    44.1 kHz mono vocals stem of cut/audio.wav (separate stage)
Piece and clip times are on the cut timeline (= the vocals). Clip ids are `<video_id>_<idx:04d>`.

Stage placement (spec 4.1, with the LLM as a GPU tenant): the pool runs download, cut, pieces, group, select and
the final bookkeeping; the GPU thread runs GPU_STAGES stage-major: it takes the first GPU stage in STAGES order
that has ready videos and runs it for all of them before moving on, so each model loads once per batch. The
punctuation LLM is a local Ollama model that also needs VRAM, so its stages run on the GPU thread too: the torch
model is released before an LLM batch, and Ollama is told to unload its model after it.
A stage is marked done only after its outputs are persisted, so a restart reruns unfinished stages.

Studio videos (origin="studio") start as `fetching`, run `download` only and stop at `awaiting_cut`;
`start_video` writes the keep ranges and queues the rest. Dashboard videos run straight through.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .gpu import run_with_oom_retry
from .pipeline import STAGES
from .pipeline.group import group_pieces
from .pipeline.routing import route
from .pipeline.select import pick, score, select_clip

GPU_STAGES = ("separate", "transcribe_a", "transcribe_b", "punctuate_pieces", "quality", "punctuate")
LLM_STAGES = ("punctuate_pieces", "punctuate")  # GPU-thread stages whose GPU tenant is the Ollama model
FINALIZE = "finalize"
PUNCT_RETRY = "punctuate_retry"
REGROUP_STAGES = ("punctuate_pieces", "group", "quality", "select", "punctuate")  # rerun by a regroup
RECOVER_S = 0.2  # pause after an unexpected worker error, before trying again
POOL_WORKERS = 2
MIN_CLEAN_S = 30.0
WARN_SHORT = "<30 s clean audio"
WARN_POOR_STT = "poor STT"

log = logging.getLogger(__name__)

_ACTIVE = ("fetching", "queued", "running")  # `fetching` only while download is not done
_QUALITY_FIELDS = ("dnsmos", "speaker_sim", "drop_reason", "status")
_PIECE_KEYS = ("idx", "range_idx", "start_s", "end_s", "text_a", "text_b", "b_tokens", "b_compression",
               "b_retry_failed", "text", "broken", "no_punct")
_SELECT_FIELDS = ("text_a", "text_b", "score_a", "score_b", "text_pick", "cer", "flag", "status", "drop_reason")
_PUNCT_STATUSES = ["auto", "flagged", "needs_review"]


def kept_text(c: dict) -> str:
    """The clip's machine text: the model `text_pick` names (legacy clips without a pick: model A)."""
    return (c["text_b"] if c.get("text_pick") == "b" else c["text_a"]) or ""


def _machine_text(c: dict) -> bool:
    """True while text_final is untouched by a reviewer: not set yet, or the unpunctuated kept STT text."""
    return c["text_final"] is None or c["text_final"] == kept_text(c)


def video_dir(data_dir: Path, dataset_id: str, video_id: str) -> Path:
    return Path(data_dir) / "videos" / dataset_id / video_id


def audio_path(data_dir: Path, dataset_id: str, video_id: str) -> Path:
    return video_dir(data_dir, dataset_id, video_id) / "download" / "audio.wav"


def cut_dir(data_dir: Path, dataset_id: str, video_id: str) -> Path:
    return video_dir(data_dir, dataset_id, video_id) / "cut"


def vocals_path(data_dir: Path, dataset_id: str, video_id: str) -> Path:
    return video_dir(data_dir, dataset_id, video_id) / "separate" / "vocals.wav"


def _transcribe_b(pieces: list[dict], vocals_wav: Path, model, on_progress: Callable[[int, int], None]) -> list[dict]:
    """Default transcribe_b: the vocals at 16 kHz through the loaded model B (one batched decode of all pieces)."""
    if not pieces:
        return []
    from .pipeline.audio16k import load_16k

    out = model(load_16k(vocals_wav), pieces)
    on_progress(len(pieces), len(pieces))
    return out


def _llm_unload(llm) -> None:
    """Default llm_unload: ask the LLM client to free its VRAM (OllamaClient.unload); other clients: nothing."""
    unload = getattr(llm, "unload", None)
    if unload is not None:
        unload()


@dataclass
class StageFns:
    """The stage callables, with the real stage modules' signatures (injectable for tests)."""

    download: Callable[..., dict]       # (video_id, url, out_dir, on_progress) -> metadata
    cut: Callable[..., list]            # (audio_wav, out_dir, cuts) -> time map
    separate: Callable[..., None]       # (audio_wav, out_wav, model, segment_s, on_progress)
    pieces: Callable[..., list]         # (vocals_wav, cmap, th) -> [{idx, range_idx, start_s, end_s}]
    transcribe_a: Callable[..., list]   # (pieces, vocals_wav, model, on_progress) -> [text]
    transcribe_b: Callable[..., list]   # (pieces, vocals_wav, model, on_progress) -> [{text, tokens, compression, retry_failed}]
    quality: Callable[..., list]        # (vocals_wav, clips, th, embedder=) -> clips
    punctuate: Callable[..., tuple]     # (text, llm) -> (text, no_punct)
    llm_unload: Callable[[Any], None] = field(default=_llm_unload)  # (llm): free the LLM's VRAM

    @classmethod
    def default(cls) -> StageFns:
        from .pipeline.cut import cut_audio
        from .pipeline.download import download
        from .pipeline.pieces import pieces
        from .pipeline.punct import punctuate
        from .pipeline.quality import quality
        from .pipeline.separate import separate
        from .pipeline.transcribe import transcribe_pieces_a

        return cls(download=download, cut=cut_audio, separate=separate, pieces=pieces,
                   transcribe_a=transcribe_pieces_a, transcribe_b=_transcribe_b, quality=quality,
                   punctuate=punctuate, llm_unload=_llm_unload)


class StageCancelled(Exception):
    """Raised from a progress callback when the user stopped the video while its stage was running."""


class JobRunner:
    def __init__(self, store, bus, gpu, llm, settings, stages: StageFns | None = None):
        self.store, self.bus, self.gpu, self.llm, self.settings = store, bus, gpu, llm, settings
        self.stages = stages or StageFns.default()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._tasks: queue.Queue = queue.Queue()
        self._active: set[str] = set()          # dataset ids being scheduled
        self._inflight: set[tuple] = set()      # (dataset_id, video_id[, PUNCT_RETRY])
        self._cancel: set[tuple[str, str]] = set()  # stopped videos whose running stage must end
        self._llm_retry: list[str] = []         # dataset ids waiting for the GPU thread
        self._llm_busy = 0
        self._t0: dict[tuple, float] = {}
        self._threads: list[threading.Thread] = []

    # --- public API
    def start(self) -> None:
        """Resume after a restart: `running` goes back to `queued`; `fetching` resumes its download (or, when the
        download finished just before the restart, moves on to `awaiting_cut`); `awaiting_cut` is left alone."""
        for ds in self.store.list_datasets():
            for v in self.store.list_videos(ds["id"]):
                if v["status"] == "running":
                    self.store.update_video(ds["id"], v["id"], status="queued")
                elif v["status"] == "fetching" and self.store.stage_done(ds["id"], v["id"], "download"):
                    self.store.update_video(ds["id"], v["id"], status="awaiting_cut")
                    continue
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

    def start_video(self, dataset_id: str, video_id: str, cuts: list[list[float]] | None) -> None:
        """Studio: the user chose the keep ranges (None or [] = whole video); the video runs from `cut` on."""
        with self._lock:
            self._cancel.discard((dataset_id, video_id))
        # always non-NULL for a started video: NULL means "the user never chose ranges" (see retry)
        self.store.update_video(dataset_id, video_id, cuts=json.dumps(cuts if cuts is not None else []),
                                status="queued", error=None)
        self.enqueue(dataset_id)

    def retry(self, dataset_id: str, video_id: str) -> None:
        """Rerun from the first unfinished stage. A Studio video goes back to `fetching` when its download is
        unfinished, and to `awaiting_cut` (not queued) when the user never started it with ranges."""
        with self._lock:
            self._cancel.discard((dataset_id, video_id))
        step = self._next_step(dataset_id, video_id)
        if step != FINALIZE:
            self.store.mark_stage(dataset_id, video_id, step, False)
        v = self._video(dataset_id, video_id)
        studio = v is not None and v.get("origin") == "studio"
        if studio and step == "cut" and v.get("cuts") is None:
            self.store.update_video(dataset_id, video_id, status="awaiting_cut", error=None)
            return
        self.store.update_video(dataset_id, video_id, status="fetching" if studio and step == "download" else "queued",
                                error=None)
        self.enqueue(dataset_id)

    def busy(self, dataset_id: str, video_id: str) -> bool:
        """True while the video is queued for, or running, a stage or an LLM retry in this runner."""
        with self._lock:
            return any(k[:2] == (dataset_id, video_id) for k in self._inflight)

    def skip(self, dataset_id: str, video_id: str) -> None:
        """Stop the video: no further stages, and a running stage ends at its next progress report
        (stages without progress reports finish first, and their result is discarded)."""
        self.store.update_video(dataset_id, video_id, status="skipped")
        with self._lock:
            if any(k[:2] == (dataset_id, video_id) for k in self._inflight):
                self._cancel.add((dataset_id, video_id))

    def wait_free(self, dataset_id: str, video_id: str, timeout: float) -> bool:
        """Wait until the video has nothing queued or running; True when it is free."""
        deadline = time.monotonic() + timeout
        while self.busy(dataset_id, video_id):
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return True

    def _cancelled(self, ds: str, vid: str) -> bool:
        with self._lock:
            return (ds, vid) in self._cancel

    def retry_llm(self, dataset_id: str) -> None:
        """For videos whose stages already ran: regroup videos with `unsplit` clips that nobody reviewed
        (punctuate_pieces, group, quality, select, punctuate), and punctuate no_punct clips of the others."""
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
    def _schedulable(self, ds: str, v: dict) -> bool:
        """Queued/running videos, and `fetching` ones until their download is done."""
        if v["status"] in ("queued", "running"):
            return True
        return v["status"] == "fetching" and not self.store.stage_done(ds, v["id"], "download")

    def _idle(self) -> bool:
        with self._lock:
            if self._inflight or self._llm_retry or self._llm_busy:
                return False
            active = list(self._active)
        return not any(self._schedulable(ds, v) for ds in active for v in self.store.list_videos(ds))

    def _next_step(self, ds: str, vid: str) -> str:
        for s in STAGES:
            if not self.store.stage_done(ds, vid, s):
                return s
        return FINALIZE

    def _ready(self) -> list[tuple[str, str, str]]:
        """(dataset_id, video_id, step) for schedulable videos not in flight. Caller holds the lock."""
        out = []
        for ds in sorted(self._active):
            for v in self.store.list_videos(ds):
                if (ds, v["id"]) not in self._inflight and self._schedulable(ds, v):
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
            try:
                fn = self._finalize if step == FINALIZE else getattr(self, f"_stage_{step}")
                self._run_step(ds, vid, step, fn)
            except Exception as e:  # top-level guard: a worker thread must never die
                self._worker_error("pool", e, ds, vid)
                with self._lock:
                    self._inflight.discard((ds, vid))
                self._fail(ds, vid, step, e)
                self._recover()

    def _gpu_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._gpu_tick()
            except Exception as e:  # top-level guard: the GPU thread must never die
                self._worker_error("gpu", e)
                self._recover()

    def _gpu_tick(self) -> None:
        with self._cv:
            retry_ds = self._llm_retry.pop(0) if self._llm_retry else None
            if retry_ds is not None:
                self._llm_busy += 1
            batch = None if retry_ds is not None else self._gpu_batch()
            if retry_ds is None and batch is None:
                self._cv.wait(0.5)
                return
        if retry_ds is not None:
            try:
                self._run_llm_retry(retry_ds)
            finally:
                with self._lock:
                    self._llm_busy -= 1
            return
        stage, vids = batch
        started = 0
        llm = stage in LLM_STAGES
        try:
            if llm:
                self.gpu.release()  # the Ollama model needs the VRAM the torch model holds
            for ds, vid in vids:
                if self._stop.is_set():
                    break
                started += 1
                self._run_step(ds, vid, stage, getattr(self, f"_stage_{stage}"))
        finally:
            if llm:
                self._unload_llm()  # before the next torch/CT2 model loads
            with self._lock:  # videos of the batch that never started (stop or an unexpected error)
                self._inflight.difference_update(vids[started:])

    def _gpu_batch(self) -> tuple[str, list[tuple[str, str]]] | None:
        """First GPU stage in STAGES order with ready videos, and all of them. Caller holds the lock."""
        ready = [r for r in self._ready() if r[2] in GPU_STAGES]
        for stage in GPU_STAGES:
            vids = [(ds, vid) for ds, vid, step in ready if step == stage]
            if vids:
                self._inflight.update(vids)
                return stage, vids
        return None

    def _unload_llm(self) -> None:
        try:
            self.stages.llm_unload(self.llm)
        except Exception as e:  # a failed unload must not fail the batch
            self._worker_error("llm unload", e)

    def _with_llm(self, fn: Callable[[], None]) -> None:
        """Run LLM work on the GPU thread: release the torch model first, unload the Ollama model after."""
        self.gpu.release()
        try:
            fn()
        finally:
            self._unload_llm()

    def _run_step(self, ds: str, vid: str, step: str, fn: Callable[[str, str], None]) -> None:
        try:
            status = self._status(ds, vid)
            if status not in _ACTIVE:
                return  # skipped while waiting
            if step != FINALIZE:
                if status != "fetching":  # a Studio download shows as fetching, not running
                    self.store.update_video(ds, vid, status="running")
                self._t0[(ds, vid, step)] = time.monotonic()
            fn(ds, vid)
            if self._cancelled(ds, vid):
                raise StageCancelled()
            if step != FINALIZE:
                if self._status(ds, vid) is None:
                    return  # video deleted meanwhile: do not leave an orphan stage mark
                self.store.mark_stage(ds, vid, step, True)
                self._publish(type="stage_done", dataset_id=ds, video_id=vid, stage=step)
                if step == "download":
                    self._after_download(ds, vid)
        except Exception as e:
            if self._cancelled(ds, vid):  # stopped by the user: not a failure, whatever the stage raised
                self._publish(type="stage_cancelled", dataset_id=ds, video_id=vid, stage=step)
            else:
                self._fail(ds, vid, step, e)
        finally:
            self._t0.pop((ds, vid, step), None)
            with self._lock:
                self._inflight.discard((ds, vid))
                self._cancel.discard((ds, vid))
            self._safe_kick()

    def _after_download(self, ds: str, vid: str) -> None:
        """A Studio video waits for the user's keep ranges (marked after the download mark, so a restart in
        between still finds the download done; start() then moves it on)."""
        v = self._video(ds, vid)
        if v is not None and v.get("origin") == "studio" and v["status"] in ("fetching", "running"):
            self.store.update_video(ds, vid, status="awaiting_cut")
            self._publish(type="awaiting_cut", dataset_id=ds, video_id=vid)

    def _fail(self, ds: str, vid: str, step: str, err: Exception) -> None:
        """Mark the video failed (unless skipped or deleted) and publish stage_failed. A store error here is
        retried once and then logged; it never propagates."""
        for attempt in range(2):
            try:
                if self._status(ds, vid) not in ("skipped", None):
                    self.store.update_video(ds, vid, status="failed", error=str(err))
                break
            except Exception as e:
                self._worker_error("mark failed", e, ds, vid)
                if attempt == 0:
                    time.sleep(RECOVER_S)
        self._publish(type="stage_failed", dataset_id=ds, video_id=vid, stage=step, error=str(err))

    def _worker_error(self, where: str, err: Exception, ds: str | None = None, vid: str | None = None) -> None:
        log.error("ytds %s worker error (dataset=%s video=%s)", where, ds, vid, exc_info=err)
        self._publish(type="log", dataset_id=ds, video_id=vid, message=f"internal error in {where}: {err!r}")

    def _safe_kick(self) -> None:
        try:
            self._kick()
        except Exception as e:
            self._worker_error("scheduler", e)

    def _recover(self) -> None:
        """After an unexpected error: pause briefly, then reschedule so no ready video is left unqueued."""
        time.sleep(RECOVER_S)
        self._safe_kick()

    # --- helpers
    def _video(self, ds: str, vid: str) -> dict | None:
        return next((v for v in self.store.list_videos(ds) if v["id"] == vid), None)

    def _status(self, ds: str, vid: str) -> str | None:
        v = self._video(ds, vid)
        return v["status"] if v else None

    def _publish(self, **event: Any) -> None:
        try:
            self.bus.publish(event)
        except Exception:
            pass  # a closed event loop must not fail a stage

    def _progress(self, ds: str, vid: str, stage: str) -> Callable[[int, int], None]:
        def cb(done: int, total: int) -> None:
            if self._cancelled(ds, vid):
                raise StageCancelled()
            eta = None
            t0 = self._t0.get((ds, vid, stage))
            if t0 is not None and done > 0 and total > done:
                eta = (time.monotonic() - t0) / done * (total - done)
            self._publish(type="progress", dataset_id=ds, video_id=vid, stage=stage,
                          done=done, total=total, eta_s=eta)
        return cb

    def _vocals(self, ds: str, vid: str) -> Path:
        return vocals_path(self.settings.data_dir, ds, vid)

    def _save(self, ds: str, clips: list[dict], fields: tuple[str, ...]) -> None:
        for c in clips:
            self.store.update_clip(ds, c["id"], **{k: c.get(k) for k in fields})

    def _write_pieces(self, ds: str, vid: str, rows: list[dict]) -> None:
        """All pieces of the video in one transaction (a stage's piece outputs are persisted together)."""
        self.store.replace_pieces(ds, vid, [{k: p.get(k) for k in _PIECE_KEYS} for p in rows])

    # --- stages
    def _stage_download(self, ds: str, vid: str) -> None:
        url = self._video(ds, vid)["url"]
        audio = audio_path(self.settings.data_dir, ds, vid)
        meta = self.stages.download(vid, url, audio.parent, self._progress(ds, vid, "download"))
        self.store.update_video(ds, vid, **meta)

    def _stage_cut(self, ds: str, vid: str) -> None:
        raw = self._video(ds, vid).get("cuts")
        cuts = json.loads(raw) if raw else None
        out = cut_dir(self.settings.data_dir, ds, vid)
        cmap = self.stages.cut(audio_path(self.settings.data_dir, ds, vid), out, cuts)
        out.mkdir(parents=True, exist_ok=True)
        (out / "map.json").write_text(json.dumps(cmap), encoding="utf-8")  # what pieces and the API read

    def _stage_separate(self, ds: str, vid: str) -> None:
        audio = cut_dir(self.settings.data_dir, ds, vid) / "audio.wav"
        vocals = self._vocals(ds, vid)
        vocals.parent.mkdir(parents=True, exist_ok=True)
        cb = self._progress(ds, vid, "separate")
        run_with_oom_retry(
            lambda seg: self.stages.separate(audio, vocals, self.gpu.use("demucs"), seg, cb),
            self.store.get_thresholds().demucs_segment_s,
        )

    def _stage_pieces(self, ds: str, vid: str) -> None:
        cmap = json.loads((cut_dir(self.settings.data_dir, ds, vid) / "map.json").read_text(encoding="utf-8"))
        ps = self.stages.pieces(self._vocals(ds, vid), cmap, self.store.get_thresholds())
        self._write_pieces(ds, vid, ps)

    def _stage_transcribe_a(self, ds: str, vid: str) -> None:
        pieces = self.store.pieces(ds, vid)
        if not pieces:
            return  # no speech: do not load model A for nothing
        model = self.gpu.use("model_a")  # once per stage run: a Settings change mid-stage does not affect it
        texts = self.stages.transcribe_a(pieces, self._vocals(ds, vid), model, self._progress(ds, vid, "transcribe_a"))
        if len(texts) != len(pieces):
            raise RuntimeError(f"model A returned {len(texts)} texts for {len(pieces)} pieces")
        self._write_pieces(ds, vid, [{**p, "text_a": t} for p, t in zip(pieces, texts)])
        self.store.update_video(ds, vid, model_a=getattr(model, "option_id", None))

    def _stage_transcribe_b(self, ds: str, vid: str) -> None:
        pieces = self.store.pieces(ds, vid)
        if not pieces:
            return
        model = self.gpu.use("model_b")
        outs = self.stages.transcribe_b(pieces, self._vocals(ds, vid), model, self._progress(ds, vid, "transcribe_b"))
        if len(outs) != len(pieces):
            raise RuntimeError(f"model B returned {len(outs)} results for {len(pieces)} pieces")
        self._write_pieces(ds, vid, [
            {**p, "text_b": o["text"], "b_tokens": o["tokens"], "b_compression": o["compression"],
             "b_retry_failed": int(bool(o["retry_failed"]))} for p, o in zip(pieces, outs)
        ])
        self.store.update_video(ds, vid, model_b=getattr(model, "option_id", None))

    def _stage_punctuate_pieces(self, ds: str, vid: str) -> None:
        """Per piece: keep the better of the A/B texts (spec 4.6), mark it broken when both are, and punctuate
        the kept text of each non-broken piece, in order."""
        th = self.store.get_thresholds()
        pieces = self.store.pieces(ds, vid)
        cb = self._progress(ds, vid, "punctuate_pieces")
        out = []
        for i, p in enumerate(pieces, 1):
            dur = p["end_s"] - p["start_s"]
            a, b = p["text_a"] or "", p["text_b"] or ""
            b_stats = {"max_compression": p["b_compression"] or 0.0, "retry_failed": bool(p["b_retry_failed"])}
            kept = a if pick(score(a, dur, th), score(b, dur, th, b_stats), th) == "a" else b
            broken = route(a, dur, th) == "broken" and route(b, dur, th) == "broken"
            if broken:
                text, no_punct = kept, False
            else:
                text, no_punct = self.stages.punctuate(kept, self.llm)
            out.append({**p, "text": text, "broken": int(broken), "no_punct": int(no_punct)})
            cb(i, len(pieces))
        self._write_pieces(ds, vid, out)

    def _stage_group(self, ds: str, vid: str) -> None:
        """Builds the clips. When the video already has reviewed clips (a Retry LLM regroup the reviewer got
        ahead of), they are kept: quality and select are marked done and only punctuate reruns."""
        if self._reviewed(self.store.clips(ds, video_id=vid)):
            for s in ("quality", "select"):
                self.store.mark_stage(ds, vid, s, True)
            self._publish(type="log", dataset_id=ds, video_id=vid, stage="group",
                          message="clips were reviewed before the regroup ran; kept as they are")
            return
        th = self.store.get_thresholds()
        pieces = self.store.pieces(ds, vid)
        clips: list[dict] = []
        for r in sorted({p["range_idx"] for p in pieces}):
            rp = [p for p in pieces if p["range_idx"] == r]
            punctuated = not any(p["no_punct"] for p in rp if not p["broken"])
            clips.extend(group_pieces(rp, th, punctuated))
        rows = []
        for idx, c in enumerate(clips):  # idx continuous across ranges
            rows.append({"id": f"{vid}_{idx:04d}", "idx": idx, "range_idx": c["range_idx"],
                         "start_s": c["start_s"], "end_s": c["end_s"], "flag": c["flag"],
                         "status": c["status"], "drop_reason": c["drop_reason"]})
        self.store.replace_clips(ds, vid, rows)

    def _stage_quality(self, ds: str, vid: str) -> None:
        clips = [c for c in self.store.clips(ds, video_id=vid) if c["status"] != "dropped"]
        if not clips:
            return  # no speech: do not load ECAPA for nothing
        clips = self.stages.quality(self._vocals(ds, vid), clips, self.store.get_thresholds(),
                                    embedder=self.gpu.use("ecapa"))
        self._save(ds, clips, _QUALITY_FIELDS)

    def _stage_select(self, ds: str, vid: str) -> None:
        """Clip texts from its pieces, A/B scores and pick (spec 4.6). The grouping flag `unsplit` wins over
        `disagree` (Retry LLM looks for it) and sends the clip to review; `disagree` wins over
        `incomplete_sentence`. Writes are compare-and-set, so a review made meanwhile is kept."""
        th = self.store.get_thresholds()
        clips = [c for c in self.store.clips(ds, video_id=vid) if c["status"] != "dropped"]
        all_pieces = self.store.pieces(ds, vid)
        pieces = [p for p in all_pieces if not p["broken"]]
        if clips and not all_pieces:
            raise RuntimeError("this video was segmented before the Studio pipeline and has no pieces; "
                               "delete it and add it again")
        for c in clips:
            rows = [p for p in pieces if p["range_idx"] == c["range_idx"]
                    and c["start_s"] <= (p["start_s"] + p["end_s"]) / 2 <= c["end_s"]]
            out = select_clip(rows, c["end_s"] - c["start_s"], th)
            if c["flag"] == "unsplit":
                out["flag"] = "unsplit"
                if out["status"] != "dropped":
                    out["status"] = "needs_review"
            elif out["flag"] is None:
                out["flag"] = c["flag"]
            self.store.update_clip_if(ds, c["id"], {"status": c["status"], "text_final": c["text_final"]},
                                      text_source="stt", **{k: out[k] for k in _SELECT_FIELDS})

    def _stage_punctuate(self, ds: str, vid: str, only_no_punct: bool = False) -> None:
        """Only clips whose text_final is still the machine text (NULL or the kept STT text); each write is
        compare-and-set against the snapshot, so a review edit or approval is never overwritten. Legacy
        translated clips are left as they are."""
        clips = [c for c in self.store.clips(ds, video_id=vid, statuses=_PUNCT_STATUSES)
                 if _machine_text(c) and c["text_source"] != "translated"]
        if only_no_punct:
            clips = [c for c in clips if c["no_punct"]]
        cb = self._progress(ds, vid, "punctuate")
        for i, c in enumerate(clips, 1):
            text, no_punct = self.stages.punctuate(kept_text(c), self.llm)
            self.store.update_clip_if(ds, c["id"], {"status": c["status"], "text_final": c["text_final"]},
                                      text_final=text, no_punct=int(no_punct))
            cb(i, len(clips))

    def _finalize(self, ds: str, vid: str) -> None:
        clips = self.store.clips(ds, video_id=vid)
        live = [c for c in clips if c["status"] != "dropped"]
        kept_s = sum(c["end_s"] - c["start_s"] for c in live)
        bad = [c for c in clips if c["drop_reason"] == "bad_stt"]
        warnings = []
        if kept_s < MIN_CLEAN_S:
            warnings.append(WARN_SHORT)
        if clips and len(bad) > len(clips) / 2:
            warnings.append(WARN_POOR_STT)
        warning = "; ".join(warnings) or None
        self.store.update_video(ds, vid, status="done", warning=warning)
        self._publish(type="video_done", dataset_id=ds, video_id=vid, message=warning)

    # --- LLM retry (GPU thread)
    def _reviewed(self, clips: list[dict]) -> bool:
        return any(c["status"] in ("approved", "rejected") or not _machine_text(c) for c in clips)

    def _run_llm_retry(self, ds: str) -> None:
        """A regroup (spec 4.4) clears the marks of punctuate_pieces..punctuate and queues the video, so the
        stage graph reruns them with its usual resume, failure status and GPU tenancy. Errors of the
        punctuation-only retry are logged per video and leave its status alone: its stages already finished."""
        for v in self.store.list_videos(ds):
            vid = v["id"]
            if not self.store.stage_done(ds, vid, "punctuate"):
                continue
            try:
                clips = self.store.clips(ds, video_id=vid)
                if (v["status"] == "done" and any(c["flag"] == "unsplit" for c in clips)
                        and not self._reviewed(clips)):
                    self._schedule_regroup(ds, vid)
                else:
                    self._guarded(ds, vid, PUNCT_RETRY,
                                  lambda: self._with_llm(lambda: self._stage_punctuate(ds, vid, only_no_punct=True)))
            except Exception as e:
                self._publish(type="log", dataset_id=ds, video_id=vid, stage="punctuate",
                              message=f"LLM retry failed: {e}")

    def _guarded(self, ds: str, vid: str, tag: str, fn: Callable[[], None]) -> None:
        with self._lock:
            self._inflight.add((ds, vid, tag))
        try:
            fn()
        finally:
            with self._lock:
                self._inflight.discard((ds, vid, tag))

    def _schedule_regroup(self, ds: str, vid: str) -> None:
        for stage in REGROUP_STAGES:
            self.store.mark_stage(ds, vid, stage, False)
        self.store.update_video(ds, vid, status="queued", error=None)
        self.enqueue(ds)
