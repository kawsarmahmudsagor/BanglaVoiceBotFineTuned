"""`dnsmos` step: DNSMOS P.808 / SIG / BAK / OVRL per clip (spec 6.1).

Scores come from torchmetrics' `deep_noise_suppression_mean_opinion_score` at 16 kHz with
`personalized=False`. The model is loaded lazily; tests inject a `scorer`.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..workdir import WorkDir
from ._runner import run_step

STEP = "dnsmos"
VERSION = 1
SCORES = ["dnsmos_p808", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl"]
COLUMNS = ["id", *SCORES, "error"]
SAMPLE_RATE = 16000

# A scorer maps a mono float32 16 kHz waveform to (p808, sig, bak, ovrl).
Scorer = Callable[[np.ndarray], "tuple[float, float, float, float]"]
_scorer: Scorer | None = None


def make_scorer() -> Scorer:
    """The real torchmetrics scorer (CPU or GPU). Imports torch lazily."""
    import torch
    from torchmetrics.functional.audio import deep_noise_suppression_mean_opinion_score

    def scorer(y16k: np.ndarray):
        out = deep_noise_suppression_mean_opinion_score(torch.from_numpy(np.ascontiguousarray(y16k)), SAMPLE_RATE, False)
        p808, sig, bak, ovrl = (float(v) for v in out.reshape(-1)[:4])  # order: p808, sig, bak, ovrl
        return p808, sig, bak, ovrl

    return scorer


def get_scorer() -> Scorer:
    global _scorer
    if _scorer is None:
        _scorer = make_scorer()
    return _scorer


def score_dnsmos(y16k: np.ndarray, *, scorer: Scorer | None = None) -> dict:
    scorer = scorer or get_scorer()
    values = scorer(y16k)
    return {name: float(v) for name, v in zip(SCORES, values, strict=True)}


def _load_16k(path: Path) -> np.ndarray:
    import librosa

    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(y) == 0:
        raise ValueError("empty audio")
    return y


def _row(row: dict) -> dict:
    return {"id": row["id"], **score_dnsmos(_load_16k(Path(row["src_path"])))}


def run(wd: WorkDir, *, force: bool = False, retry_errors: bool = False) -> int:
    get_scorer()  # fail fast on a missing extra instead of recording an error on every row
    # One process: the model lives on the GPU.
    return run_step(
        wd, step=STEP, version=VERSION, columns=COLUMNS, row_fn=_row, input_cols=["id", "src_path"],
        dtypes={name: "float64" for name in SCORES}, workers=1, force=force, retry_errors=retry_errors,
    )


def _cmd(args: argparse.Namespace) -> int:
    try:
        run(WorkDir(args.work), force=args.force, retry_errors=args.retry_errors)
    except (RuntimeError, ImportError) as e:
        print(f"dnsmos: cannot load the scorer: {e}")
        return 1
    return 0


def add_parser(commands) -> None:
    p = commands.add_parser("dnsmos", help="DNSMOS P.808/SIG/BAK/OVRL per clip (needs the score extra)")
    p.add_argument("--work", type=Path, default=Path("var/dataprep/bengaliai"), help="work dir")
    p.add_argument("--force", action="store_true", help="recompute all rows")
    p.add_argument("--retry-errors", action="store_true", help="recompute rows whose error is set")
    p.set_defaults(func=_cmd)
