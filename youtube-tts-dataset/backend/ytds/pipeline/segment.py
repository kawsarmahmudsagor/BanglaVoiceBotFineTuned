"""Segment stage: Silero VAD regions -> clip plan within [min_s, max_s]."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import Thresholds
from .audio16k import SR, load_16k

_vad_model = None


def _get_vad():
    global _vad_model
    if _vad_model is None:
        from silero_vad import load_silero_vad

        _vad_model = load_silero_vad()
    return _vad_model


def vad_regions(vocals_16k: np.ndarray) -> list[tuple[float, float]]:
    import torch
    from silero_vad import get_speech_timestamps

    ts = get_speech_timestamps(
        torch.from_numpy(np.ascontiguousarray(vocals_16k)), _get_vad(),
        sampling_rate=SR, return_seconds=True,
    )
    return [(float(t["start"]), float(t["end"])) for t in ts]


def _split(span: list[tuple[float, float]], max_s: float) -> list[tuple[float, float]]:
    """span: original regions forming one merged span."""
    start, end = span[0][0], span[-1][1]
    if end - start <= max_s:
        return [(start, end)]
    if len(span) == 1:
        out, t = [], start
        while t < end:
            out.append((t, min(t + max_s, end)))
            t += max_s
        return out
    k = max(range(1, len(span)), key=lambda i: span[i][0] - span[i - 1][1])
    return _split(span[:k], max_s) + _split(span[k:], max_s)


def plan_clips(
    regions: list[tuple[float, float]], *, merge_gap_s: float, min_s: float, max_s: float
) -> list[tuple[float, float]]:
    regions = sorted(regions)
    spans: list[list[tuple[float, float]]] = []
    for r in regions:
        if spans and r[0] - spans[-1][-1][1] < merge_gap_s:
            spans[-1].append(r)
        else:
            spans.append([r])
    pieces: list[tuple[float, float]] = []
    for sp in spans:
        pieces.extend(_split(sp, max_s))

    i = 0
    while i < len(pieces):
        s, e = pieces[i]
        if e - s >= min_s:
            i += 1
            continue
        gap_prev = s - pieces[i - 1][1] if i > 0 else float("inf")
        gap_next = pieces[i + 1][0] - e if i + 1 < len(pieces) else float("inf")
        if gap_prev == gap_next == float("inf"):
            del pieces[i]
            continue
        j = i - 1 if gap_prev <= gap_next else i + 1
        a, b = sorted((i, j))
        merged = (pieces[a][0], pieces[b][1])
        if merged[1] - merged[0] <= max_s:
            pieces[a:b + 1] = [merged]
            i = a  # re-check the merged piece
        else:
            del pieces[i]
    return pieces


def segment(vocals_wav: Path, th: Thresholds) -> list[dict]:
    regions = vad_regions(load_16k(vocals_wav))
    spans = plan_clips(regions, merge_gap_s=th.merge_gap_s, min_s=th.clip_min_s, max_s=th.clip_max_s)
    return [{"idx": i, "start_s": s, "end_s": e} for i, (s, e) in enumerate(spans)]
