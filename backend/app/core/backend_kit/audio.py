"""Audio helpers (needs the `audio` extra: numpy, soundfile, soxr). Not imported by the package __init__."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr


def load_audio(path: str | Path, sr: int) -> np.ndarray:
    """Mono float32 at `sr`. soundfile first; formats it cannot open (m4a, aac, webm, ...) go through
    ffmpeg when it is on PATH. Without ffmpeg the soundfile error is raised unchanged."""
    try:
        data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError):
        if shutil.which("ffmpeg") is None:
            raise
        return _ffmpeg_decode(path, sr)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if file_sr != sr:
        mono = soxr.resample(mono, file_sr, sr)
    return np.ascontiguousarray(mono, dtype=np.float32)


class FfmpegDecodeError(RuntimeError):
    pass


def _ffmpeg_decode(path: str | Path, sr: int) -> np.ndarray:
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout:
        msg = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise FfmpegDecodeError(msg[-1] if msg else f"ffmpeg exit {proc.returncode}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")


def duration_s(path: str | Path) -> float:
    return float(sf.info(str(path)).duration)


def write_flac(path: Path, audio: np.ndarray, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, format="FLAC", subtype="PCM_16")
