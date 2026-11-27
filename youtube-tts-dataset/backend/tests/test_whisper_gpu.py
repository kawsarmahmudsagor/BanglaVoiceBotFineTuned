"""Model B on the GPU: P1 (max_length 880) lifts the 224-token cap of an 18 s Bangla window (spec 4.5).

The 18 s window is real speech from the smoke video (tests/data/smoke_url.txt), 100-118 s, which needs more than
224 tokens. bn_clip.wav copies do not work for this: the fine-tune ends the decode after one sentence (121 tokens
for one, two or three copies), so it never reaches the cap.

Run on the RTX 4060 with: uv run pytest -m gpu -q -s tests/test_whisper_gpu.py
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from ytds.config import load_settings
from ytds.pipeline.audio16k import SR, load_16k

SMOKE_URL = (Path(__file__).parent / "data" / "smoke_url.txt").read_text(encoding="utf-8").strip()
WINDOW = (100.0, 118.0)  # seconds of the smoke video: continuous speech, 264 tokens with max_length 880


def _gpu_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout
    return int(out.splitlines()[0].strip())


@pytest.mark.gpu
@pytest.mark.network
def test_max_length_880_lifts_the_224_token_cap(tmp_path):
    import torch  # noqa: F401  as in load_whisper: makes torch/lib's cuBLAS DLL loadable for CTranslate2
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    from ytds.pipeline.download import download, expand_urls
    from ytds.pipeline.whisper_b import Window, _decode

    [(vid, url)] = expand_urls(SMOKE_URL)
    download(vid, url, tmp_path, lambda done, total: None)
    audio = load_16k(tmp_path / "audio.wav")[int(WINDOW[0] * SR):int(WINDOW[1] * SR)]
    assert len(audio) == 18 * SR

    path = load_settings().model_b_path
    baseline = _gpu_used_mib()
    t0 = time.monotonic()
    model = WhisperModel(path, device="cuda", compute_type="float16")
    pipeline = BatchedInferencePipeline(model=model)
    window = [Window(0, len(audio))]
    try:
        model.max_length = 448  # faster-whisper's default: CT2 stops at 224 generated tokens
        short = _decode(pipeline, audio, window)[0]
        used_448 = _gpu_used_mib()
        model.max_length = 880  # P1
        long = _decode(pipeline, audio, window)[0]
        used_880 = _gpu_used_mib()
    finally:
        del pipeline, model
    print(f"\nmodel B: {path}")
    print(f"max_length 448: {short.tokens} tokens, compression {short.compression:.2f}: {short.text}")
    print(f"max_length 880: {long.tokens} tokens, compression {long.compression:.2f}: {long.text}")
    print(f"load + 2 decodes: {time.monotonic() - t0:.1f} s; nvidia-smi baseline {baseline} MiB, "
          f"after 448 decode {used_448} MiB, after 880 decode {used_880} MiB")
    assert short.tokens == 224
    assert 224 < long.tokens < 440
