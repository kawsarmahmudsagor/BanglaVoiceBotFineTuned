"""Pieces stage: Silero VAD regions -> speech pieces of at most piece_max_s, never crossing a kept range."""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from ..config import Thresholds
from .audio16k import SR, load_16k
from .cut import range_spans

MIN_PIECE_S = 1.0  # a shorter piece is a lone word: join it to its neighbour when that fits
SPLIT_SEARCH_S = 2.0
FRAME = SR // 50  # 20 ms

_vad_model = None
_vad_lock = threading.Lock()  # silero model is stateful; also guards lazy init


def _get_vad():
    global _vad_model
    if _vad_model is None:
        from silero_vad import load_silero_vad

        _vad_model = load_silero_vad()
    return _vad_model


def vad_regions(vocals_16k: np.ndarray) -> list[tuple[float, float]]:
    import torch
    from silero_vad import get_speech_timestamps

    wav = torch.from_numpy(np.ascontiguousarray(vocals_16k))
    with _vad_lock:
        ts = get_speech_timestamps(wav, _get_vad(), sampling_rate=SR, return_seconds=True)
    return [(float(t["start"]), float(t["end"])) for t in ts]


def _quietest_split(audio: np.ndarray, lo_s: float, hi_s: float) -> float:
    """Time (s) of the lowest-energy 20 ms frame centre in [lo_s, hi_s): most likely a gap between words."""
    lo = max(int(lo_s * SR), 0)
    hi = min(int(hi_s * SR), len(audio))
    seg = audio[lo:hi]
    n = len(seg) // FRAME
    if n < 2:
        return hi_s
    energy = (seg[: n * FRAME].astype("float64").reshape(n, FRAME) ** 2).mean(axis=1)
    return (lo + int(np.argmin(energy)) * FRAME + FRAME // 2) / SR


def plan_pieces(
    regions: list[tuple[float, float]],
    spans: list[tuple[int, float, float]],
    audio16k: np.ndarray,
    max_s: float,
    gap_s: float,
) -> list[dict]:
    out: list[dict] = []
    for range_idx, span_s, span_e in spans:
        rs = sorted(
            (max(s, span_s), min(e, span_e)) for s, e in regions if min(e, span_e) > max(s, span_s)
        )
        cur: tuple[float, float] | None = None

        def flush():
            nonlocal cur
            if cur:
                out.append({"range_idx": range_idx, "start_s": cur[0], "end_s": cur[1]})
            cur = None

        for s, e in rs:
            while e - s > max_s:  # unbroken speech longer than a piece
                flush()
                cut = _quietest_split(audio16k, s + max_s - SPLIT_SEARCH_S, s + max_s)
                cut = min(max(cut, s + 1e-3), s + max_s)
                out.append({"range_idx": range_idx, "start_s": s, "end_s": cut})
                s = cut
            if cur is None:
                cur = (s, e)
                continue
            gap = s - cur[1]
            fits = e - cur[0] <= max_s
            if fits and (gap < gap_s or cur[1] - cur[0] < MIN_PIECE_S):
                cur = (cur[0], e)
            else:
                flush()
                cur = (s, e)
        flush()
    return [{"idx": i, **p} for i, p in enumerate(out)]


def pieces(vocals_wav: Path, cmap: list[dict], th: Thresholds) -> list[dict]:
    y = load_16k(vocals_wav)
    return plan_pieces(vad_regions(y), range_spans(cmap), y, th.piece_max_s, th.piece_gap_s)
