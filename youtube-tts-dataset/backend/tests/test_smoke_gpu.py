"""Heavy smoke test: one short public Bangla video through the real pipeline to export on the GPU.

Run on the RTX 4060 with: uv run pytest -m "gpu and network" -v -s tests/test_smoke_gpu.py
"""
from __future__ import annotations

import csv
import subprocess
import threading
import time
from pathlib import Path

import pytest
import torch

from ytds.api import _default_runner
from ytds.config import load_settings
from ytds.events import EventBus
from ytds.exporter import export_dataset
from ytds.pipeline.download import expand_urls
from ytds.store import Store

SMOKE_URL = (Path(__file__).parent / "data" / "smoke_url.txt").read_text(encoding="utf-8").strip()
VRAM_LIMIT_MIB = 7 * 1024
RUN_TIMEOUT_S = 3600


def _gpu_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout
    return int(out.splitlines()[0].strip())


class _VramSampler(threading.Thread):
    """Samples device-wide memory.used; sees CTranslate2 allocations that torch's allocator does not."""

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


@pytest.mark.gpu
@pytest.mark.network
def test_smoke_full_pipeline_to_export(tmp_path):
    assert torch.cuda.is_available()
    settings = load_settings().model_copy(update={"data_dir": tmp_path / "data", "export_dir": tmp_path / "export"})
    store = Store(settings.data_dir / "ytds.db")
    runner = _default_runner(settings, store, EventBus())

    ds = store.create_dataset("smoke", "smoke_speaker")
    pairs = expand_urls(SMOKE_URL)
    assert len(pairs) == 1
    for vid, url in pairs:
        store.add_video(ds, vid, url)

    torch.cuda.reset_peak_memory_stats()
    baseline = _gpu_used_mib()
    sampler = _VramSampler()
    sampler.start()
    t0 = time.monotonic()
    runner.start()
    try:
        runner.enqueue(ds)
        assert runner.wait_idle(RUN_TIMEOUT_S), "pipeline did not finish in time"
    finally:
        runner.stop()
        runner.gpu.release()
        pipeline_s = time.monotonic() - t0
        peak_smi = sampler.stop()

    videos = store.list_videos(ds)
    assert [v["status"] for v in videos] == ["done"], videos

    # Review: approve every reviewable clip through the store. pending_llm clips stay when the LLM is down.
    statuses: dict[str, int] = {}
    for c in store.clips(ds):
        statuses[c["status"]] = statuses.get(c["status"], 0) + 1
        if c["status"] == "flagged" or (c["status"] == "needs_review" and (c["text_final"] or "").strip()):
            store.update_clip(ds, c["id"], status="approved")

    out_dir = tmp_path / "export" / "smoke"
    t1 = time.monotonic()
    summary = export_dataset(store, ds, out_dir, lambda done, total: None,
                             data_dir=settings.data_dir, skip_unreviewed_translated=True)
    export_s = time.monotonic() - t1

    with open(out_dir / "metadata.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) >= 1

    from app.datasets.validate_audio import validate_audio

    report, _ = validate_audio(out_dir, "tts")
    assert report.errors == []

    peak_torch = torch.cuda.max_memory_allocated()
    no_punct = sum(1 for c in store.clips(ds) if c["no_punct"])
    print(f"\nclip statuses after pipeline: {statuses}; no_punct: {no_punct}")
    print(f"export summary: {summary}")
    print(f"pipeline wall time: {pipeline_s:.1f} s; export wall time: {export_s:.1f} s")
    print(f"peak torch allocated: {peak_torch / 1024**2:.0f} MiB")
    print(f"nvidia-smi: baseline {baseline} MiB, peak {peak_smi} MiB, delta {peak_smi - baseline} MiB")
    assert peak_torch < 7 * 1024**3
    assert peak_smi - baseline < VRAM_LIMIT_MIB
