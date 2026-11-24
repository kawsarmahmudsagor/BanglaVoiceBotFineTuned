"""`audio-stats` step: duration, sample rate, clipping, silence share and loudness (spec 6.1)."""
from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np

from ..workdir import WorkDir
from ._runner import run_step

STEP = "audio_stats"
VERSION = 1
COLUMNS = ["id", "dur", "src_sr", "clip_frac", "silence_share", "lufs", "error"]

CLIP_LEVEL = 0.99
FRAME_S = 0.020
SILENCE_DBFS = -50.0
MIN_LUFS_S = 0.4  # pyloudnorm's gating block is 400 ms


def audio_stats(path: Path) -> dict:
    import librosa
    import pyloudnorm

    with warnings.catch_warnings():
        # librosa falls back to audioread with a warning on undecodable input; the exception is what we record.
        warnings.simplefilter("ignore")
        y, sr = librosa.load(path, sr=None, mono=True)
    n = len(y)
    dur = n / sr
    clip_frac = float(np.mean(np.abs(y) >= CLIP_LEVEL)) if n else 0.0
    frame = max(1, round(FRAME_S * sr))
    n_frames = n // frame
    if n_frames:
        rms = np.sqrt(np.mean(y[: n_frames * frame].reshape(n_frames, frame).astype("float64") ** 2, axis=1))
        silence_share = float(np.mean(rms < 10 ** (SILENCE_DBFS / 20)))
    else:
        silence_share = 0.0
    lufs = None
    if dur >= MIN_LUFS_S:
        value = float(pyloudnorm.Meter(sr).integrated_loudness(y.astype("float64")))
        lufs = value if math.isfinite(value) else None  # all-silent audio gives -inf
    return {"dur": dur, "src_sr": int(sr), "clip_frac": clip_frac, "silence_share": silence_share, "lufs": lufs}


def _row(row: dict) -> dict:
    return {"id": row["id"], **audio_stats(Path(row["src_path"]))}


def run(wd: WorkDir, *, workers: int = 1, force: bool = False, retry_errors: bool = False) -> int:
    return run_step(
        wd, step=STEP, version=VERSION, columns=COLUMNS, row_fn=_row, input_cols=["id", "src_path"],
        dtypes={"dur": "float64", "src_sr": "Int64", "clip_frac": "float64", "silence_share": "float64", "lufs": "float64"},
        workers=workers, force=force, retry_errors=retry_errors,
    )


def _cmd(args: argparse.Namespace) -> int:
    run(WorkDir(args.work), workers=args.workers, force=args.force, retry_errors=args.retry_errors)
    return 0


def add_parser(commands) -> None:
    p = commands.add_parser("audio-stats", help="compute duration, clipping, silence and loudness per clip")
    p.add_argument("--work", type=Path, default=Path("var/dataprep/bengaliai"), help="work dir")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--force", action="store_true", help="recompute all rows")
    p.add_argument("--retry-errors", action="store_true", help="recompute rows whose error is set")
    p.set_defaults(func=_cmd)
