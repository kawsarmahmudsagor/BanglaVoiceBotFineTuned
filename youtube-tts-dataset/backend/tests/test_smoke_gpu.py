"""Heavy smoke test: the real runner, models and local Ollama on the GPU, for both kinds of video at once:
- a dashboard video: the smoke URL (tests/data/smoke_url.txt), straight through to done, then exported;
- a Studio video: a local mp4 made of bn_clip.wav copies, ingested, stopped at awaiting_cut, started with one range.

Also prints (never asserts) the comparison check: per kept clip, model A on the whole clip vs the joined model A
piece texts, and how fast Ollama's VRAM is freed after the runner's unload call.

Run on the RTX 4060 with: uv run pytest -m gpu -q -s tests/test_smoke_gpu.py
"""
from __future__ import annotations

import csv
import subprocess
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf
import torch

from ytds.api import _default_runner
from ytds.config import load_settings
from ytds.events import EventBus
from ytds.exporter import export_dataset
from ytds.pipeline.audio16k import SR, load_16k
from ytds.pipeline.download import expand_urls
from ytds.pipeline.group import PAD_S
from ytds.runner import upload_file, video_dir, vocals_path
from ytds.store import Store

DATA = Path(__file__).parent / "data"
SMOKE_URL = (DATA / "smoke_url.txt").read_text(encoding="utf-8").strip()
VRAM_LIMIT_MIB = 7 * 1024
RUN_TIMEOUT_S = 3600
TERMINATORS = ("।", "?", "!", ".")
STUDIO_COPIES, STUDIO_GAP_S = 12, 0.5   # ~74 s of bn_clip.wav with pauses
STUDIO_RANGE = [2.0, 64.0]
UNLOAD_WATCH_S = 6.0
PAD_TOL = 2 * PAD_S + 0.01


def _say(text: str) -> None:
    """print that survives a cp1252 Windows console (Bangla text)."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "backslashreplace").decode("ascii"))


def _gpu_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout
    return int(out.splitlines()[0].strip())


class _VramSampler(threading.Thread):
    """Samples device-wide memory.used; sees CTranslate2 and Ollama allocations that torch's allocator does not."""

    def __init__(self, interval_s: float = 0.5):
        super().__init__(daemon=True)
        self.interval_s, self.peak, self._halt = interval_s, 0, threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            self.peak = max(self.peak, _gpu_used_mib())
            self._halt.wait(self.interval_s)

    def stop(self) -> int:
        self._halt.set()
        self.join(30)
        return self.peak


def _make_studio_mp4(path: Path) -> None:
    """bn_clip.wav copies with pauses, muxed with a small black h264 video track."""
    clip, sr = sf.read(str(DATA / "bn_clip.wav"), dtype="float32")
    gap = np.zeros(int(STUDIO_GAP_S * sr), dtype=np.float32)
    wav = path.with_suffix(".wav")
    sf.write(str(wav), np.concatenate([x for _ in range(STUDIO_COPIES) for x in (clip, gap)]), sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=320x240:r=10",
                    "-i", str(wav), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
                   check=True, timeout=120)
    wav.unlink()


def _watch_unload(runner, events: list[dict]) -> None:
    """Wrap the runner's Ollama unload: after it returns, sample nvidia-smi for UNLOAD_WATCH_S."""
    inner = runner.stages.llm_unload

    def unload(llm) -> None:
        before = _gpu_used_mib()
        t0 = time.monotonic()
        inner(llm)
        returned = time.monotonic() - t0
        samples = []
        while (t := time.monotonic() - t0) < returned + UNLOAD_WATCH_S:
            samples.append((round(t - returned, 2), _gpu_used_mib()))
            time.sleep(0.1)
        try:
            ps = [m["name"] for m in httpx.get(f"{runner.settings.ollama_host}/api/ps", timeout=5).json()["models"]]
        except Exception as e:  # report only
            ps = [f"error: {e!r}"]
        events.append({"before": before, "call_s": returned, "samples": samples, "ps_after": ps})

    runner.stages.llm_unload = unload


def _release_s(ev: dict) -> tuple[float | None, int]:
    """Seconds after the unload call returned until memory.used reached its low (within 100 MiB), and the low."""
    low = min(m for _, m in ev["samples"])
    return next((t for t, m in ev["samples"] if m <= low + 100), None), low


def _kept(clips: list[dict]) -> list[dict]:
    return [c for c in clips if c["status"] not in ("dropped", "rejected")]


@pytest.mark.gpu
@pytest.mark.network
def test_smoke_dashboard_and_studio(tmp_path):
    assert torch.cuda.is_available()
    settings = load_settings().model_copy(update={"data_dir": tmp_path / "data", "export_dir": tmp_path / "export"})
    store = Store(settings.data_dir / "ytds.db")
    runner = _default_runner(settings, store, EventBus())
    unloads: list[dict] = []
    _watch_unload(runner, unloads)

    dash = store.create_dataset("smoke", "smoke_speaker")
    studio = store.create_dataset("studio", "studio_speaker")
    pairs = expand_urls(SMOKE_URL)
    assert len(pairs) == 1

    # Studio: what the upload route does (bytes at download/upload.mp4, a `fetching` local video)
    sid = "local_smoketest1"
    up = video_dir(settings.data_dir, studio, sid) / "download" / "upload.mp4"
    up.parent.mkdir(parents=True)
    _make_studio_mp4(up)
    store.add_video(studio, sid, "local:smoke.mp4", origin="studio", source="local", status="fetching")
    store.update_video(studio, sid, title="smoke.mp4")
    assert upload_file(up.parent) == up

    torch.cuda.reset_peak_memory_stats()
    baseline = _gpu_used_mib()
    sampler = _VramSampler()
    sampler.start()
    t0 = time.monotonic()
    runner.start()
    try:
        runner.enqueue(studio)
        assert runner.wait_idle(600), "studio ingest did not finish"
        v = store.list_videos(studio)[0]
        assert v["status"] == "awaiting_cut", v
        assert v["media"] == "download/upload.mp4" and v["duration_s"] > STUDIO_RANGE[1]
        ingest_s = time.monotonic() - t0
        runner.start_video(studio, sid, [STUDIO_RANGE])
        for vid, url in pairs:
            store.add_video(dash, vid, url)
        runner.enqueue(dash)
        assert runner.wait_idle(RUN_TIMEOUT_S), "pipeline did not finish in time"
    finally:
        runner.stop()
        runner.gpu.release()
        pipeline_s = time.monotonic() - t0
        peak_smi = sampler.stop()
    peak_torch = torch.cuda.max_memory_allocated()

    for ds in (dash, studio):
        videos = store.list_videos(ds)
        assert [v["status"] for v in videos] == ["done"], videos

    # --- clip rules (spec 4.3): every kept clip 5-25 s, at least half 15-25 s, text ends a sentence or is flagged
    th = store.get_thresholds()
    stats = {}
    for name, ds in (("dashboard", dash), ("studio", studio)):
        clips = store.clips(ds)
        kept = _kept(clips)
        durs = [c["end_s"] - c["start_s"] for c in kept]
        stats[name] = (clips, kept, durs)
        _say(f"\n[{name}] {len(clips)} clips, {len(kept)} kept; durations "
             f"{[round(d, 1) for d in durs]}; statuses "
             f"{ {s: sum(c['status'] == s for c in clips) for s in {c['status'] for c in clips}} }; flags "
             f"{ {f: sum(c['flag'] == f for c in clips) for f in {c['flag'] for c in clips}} }; "
             f"no_punct {sum(1 for c in clips if c['no_punct'])}")
        for c in kept:
            _say(f"  {c['id']} {c['end_s'] - c['start_s']:.1f}s {c['status']} flag={c['flag']} "
                 f"pick={c['text_pick']} cer={c['cer']:.2f}: {c['text_final']}")
        assert kept, f"{name}: no kept clips"
        assert all(c["status"] == "needs_review" for c in kept), "nothing of the new pipeline is auto-accepted"
        hi = th.clip_max_s + PAD_TOL  # clip edges are padded by PAD_S on each side
        assert all(th.tail_min_s - 0.01 <= d <= hi for d in durs), durs
        assert sum(th.clip_min_s - 0.01 <= d <= hi for d in durs) * 2 >= len(durs), durs
        # A clip whose clip-level punctuation the word guard rejected keeps the raw STT text (no_punct), though
        # its pieces ended a sentence: it counts as marked too.
        for c in kept:
            assert (c["text_final"] or "").strip().endswith(TERMINATORS) or c["flag"] or c["no_punct"], c

    # --- comparison check (report only): model A on the whole clip vs the joined model A piece texts
    from ytds.pipeline.select import _norm, ab_cer

    torch.cuda.reset_peak_memory_stats()
    t2 = time.monotonic()
    model_a = runner.gpu.use("model_a")
    rows = []
    for name, ds in (("dashboard", dash), ("studio", studio)):
        for c in stats[name][1]:
            y = load_16k(vocals_path(settings.data_dir, ds, c["video_id"]))
            whole = model_a([y[int(c["start_s"] * SR):int(c["end_s"] * SR)]])[0]
            joined = c["text_a"] or ""
            rows.append((c["id"], ab_cer(joined, whole), len(_norm(whole)), len(_norm(joined)), whole, joined))
    runner.gpu.release()
    compare_s = time.monotonic() - t2
    compare_torch = torch.cuda.max_memory_allocated()
    _say("\ncomparison check (CER of joined pieces vs whole clip, model A):")
    for cid, cer, lw, lj, whole, joined in rows:
        _say(f"  {cid} CER {cer:.3f} chars whole {lw} joined {lj}\n    whole:  {whole}\n    joined: {joined}")
    longer = [r[0] for r in rows if r[2] > 1.10 * r[3]]
    _say(f"mean CER {np.mean([r[1] for r in rows]):.3f} over {len(rows)} clips; whole-clip text > 10 % longer: "
         f"{len(longer)} {longer}; wall time {compare_s:.1f} s; peak torch {compare_torch / 1024**2:.0f} MiB")

    # --- export of the dashboard dataset: unapproved clips block it, then every reviewable clip is approved
    from ytds.exporter import exportable

    assert exportable(store, dash) == ([], len(stats["dashboard"][1]))
    for c in store.clips(dash):
        if c["status"] == "needs_review" and (c["text_final"] or "").strip():
            store.update_clip(dash, c["id"], status="approved")
    out_dir = tmp_path / "export" / "smoke"
    t1 = time.monotonic()
    summary = export_dataset(store, dash, out_dir, lambda done, total: None, data_dir=settings.data_dir)
    export_s = time.monotonic() - t1
    with open(out_dir / "metadata.csv", encoding="utf-8", newline="") as fh:
        assert len(list(csv.DictReader(fh))) >= 1
    from app.datasets.validate_audio import validate_audio

    report, _ = validate_audio(out_dir, "tts")
    assert report.errors == []

    _say(f"\nexport summary: {summary}")
    _say(f"studio ingest {ingest_s:.1f} s; pipeline wall time {pipeline_s:.1f} s; export {export_s:.1f} s")
    _say(f"peak torch allocated: {peak_torch / 1024**2:.0f} MiB")
    _say(f"nvidia-smi: baseline {baseline} MiB, peak {peak_smi} MiB, delta {peak_smi - baseline} MiB")
    for i, ev in enumerate(unloads):
        rel, low = _release_s(ev)
        _say(f"ollama unload {i}: before {ev['before']} MiB, call took {ev['call_s']:.2f} s, low {low} MiB "
             f"reached {rel} s after it returned; /api/ps after: {ev['ps_after']}; "
             f"first samples {ev['samples'][:12]}")
    assert peak_torch < 7 * 1024**3
    assert peak_smi - baseline < VRAM_LIMIT_MIB
