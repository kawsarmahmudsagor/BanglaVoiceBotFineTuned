"""Audio helpers (needs the `audio` extra: numpy, soundfile, soxr). Not imported by the package __init__."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr


def load_audio(path: str | Path, sr: int) -> np.ndarray:
    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if file_sr != sr:
        mono = soxr.resample(mono, file_sr, sr)
    return np.ascontiguousarray(mono, dtype=np.float32)


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")


def duration_s(path: str | Path) -> float:
    return float(sf.info(str(path)).duration)
