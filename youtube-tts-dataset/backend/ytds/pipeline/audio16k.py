"""Shared helper: read a WAV as mono float32 16 kHz."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

SR = 16000


def load_16k(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != SR:
        mono = soxr.resample(mono, sr, SR)
    return np.ascontiguousarray(mono, dtype=np.float32)
