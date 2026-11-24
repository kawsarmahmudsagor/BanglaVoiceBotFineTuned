"""`brouhaha` step: SNR, C50, speech segments and longest pause per clip (spec 6.1).

The scorer maps an audio path to frame-level model output `(frame_step_s, array[n_frames, 3])` with
columns (vad, snr_db, c50_db). Everything else (segments, means, pause) is computed here, so it is
testable without the model. The default scorer loads `pyannote/brouhaha` lazily and reads the
HF token from `HF_TOKEN`.
"""
from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..workdir import WorkDir
from ._runner import run_step

STEP = "brouhaha"
VERSION = 1
COLUMNS = ["id", "snr_db", "c50_db", "max_pause_s", "speech_segments", "error"]
VAD_THRESHOLD = 0.5
MODEL = "pyannote/brouhaha"

Scorer = Callable[[Path], "tuple[float, np.ndarray]"]
_scorer: Scorer | None = None


def make_scorer() -> Scorer:
    """The real scorer: pyannote Inference over the Brouhaha model. Imports torch/pyannote lazily."""
    import torch
    from pyannote.audio import Inference, Model

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(f"HF_TOKEN is not set; {MODEL} is gated on Hugging Face")
    model = Model.from_pretrained(MODEL, use_auth_token=token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference = Inference(model, device=device)

    def scorer(path: Path):
        out = inference(str(path))
        return float(out.sliding_window.step), np.asarray(out.data, dtype="float64")

    return scorer


def segments_from_vad(vad: np.ndarray, step: float, *, threshold: float = VAD_THRESHOLD) -> list[list[float]]:
    """Runs of frames with vad >= threshold as [start, end] seconds."""
    on = np.concatenate([[False], np.asarray(vad) >= threshold, [False]])
    edges = np.flatnonzero(on[1:] != on[:-1])
    return [[float(a * step), float(b * step)] for a, b in zip(edges[::2], edges[1::2])]


def max_pause(segments: list[list[float]]) -> float:
    """Longest gap between consecutive segments; 0.0 for a single segment."""
    return max((nxt[0] - cur[1] for cur, nxt in zip(segments, segments[1:])), default=0.0)


def score_brouhaha(path: Path, *, scorer: Scorer | None = None) -> dict:
    global _scorer
    if scorer is None:
        if _scorer is None:
            _scorer = make_scorer()
        scorer = _scorer
    step, frames = scorer(Path(path))
    frames = np.asarray(frames, dtype="float64")
    segments = segments_from_vad(frames[:, 0], step)
    if not segments:
        raise ValueError("no speech")
    speech = frames[:, 0] >= VAD_THRESHOLD
    return {
        "snr_db": float(frames[speech, 1].mean()),
        "c50_db": float(frames[speech, 2].mean()),
        "max_pause_s": float(max_pause(segments)),
        "speech_segments": segments,
    }


def _row(row: dict) -> dict:
    return {"id": row["id"], **score_brouhaha(Path(row["src_path"]))}


def run(wd: WorkDir, *, force: bool = False, retry_errors: bool = False, only_profile: str | None = None) -> int:
    only_ids = None
    if only_profile:
        from ..profiles import load_profile
        from ..select import COLUMN_STEPS, eligible

        skip = frozenset(c for c, (step, _, _) in COLUMN_STEPS.items() if step == STEP)
        only_ids = set(eligible(wd, load_profile(only_profile), skip_columns=skip)["id"])
    # One process: the model lives on the GPU.
    return run_step(
        wd, step=STEP, version=VERSION, columns=COLUMNS, row_fn=_row, input_cols=["id", "src_path"],
        dtypes={"snr_db": "float64", "c50_db": "float64", "max_pause_s": "float64", "speech_segments": "object"},
        workers=1, force=force, retry_errors=retry_errors, only_ids=only_ids,
    )


def _cmd(args: argparse.Namespace) -> int:
    from ..select import SelectError

    try:
        run(WorkDir(args.work), force=args.force, retry_errors=args.retry_errors, only_profile=args.only_profile)
    except SelectError as e:
        print(f"brouhaha: {e}")
        return 1
    return 0


def add_parser(commands) -> None:
    p = commands.add_parser("brouhaha", help="SNR, C50 and speech segments per clip (needs the score extra and HF_TOKEN)")
    p.add_argument("--work", type=Path, default=Path("var/dataprep/bengaliai"), help="work dir")
    p.add_argument("--force", action="store_true", help="recompute all rows")
    p.add_argument("--retry-errors", action="store_true", help="recompute rows whose error is set")
    p.add_argument(
        "--only-profile", default=None, metavar="NAME",
        help="score only clips passing this profile's non-brouhaha filters (spec 5.2 step 2)",
    )
    p.set_defaults(func=_cmd)
