"""Deterministic train/val/test assignment: per video, val and test are each one contiguous run of clips."""
from __future__ import annotations

import random
from collections import defaultdict


def _dur(c: dict) -> float:
    return float(c["end_s"]) - float(c["start_s"])


def _run_len(durs: list[float], start: int, target: float) -> tuple[int, bool]:
    """(clips from `start` whose total duration is closest to `target`, at least one; whether the run is
    complete, i.e. not cut short by the end of the video)."""
    total, n = 0.0, 0
    for d in durs[start:]:
        if n and abs(total + d - target) >= abs(total - target):
            return n, True
        total += d
        n += 1
    return max(n, 1), total >= target


def assign_splits(
    clips: list[dict], seed: int = 0, ratios=(0.90, 0.05, 0.05), min_video_s: float = 60.0,
) -> dict[str, str]:
    """Map clip id -> "train" | "val" | "test". Videos under `min_video_s` kept seconds are all train."""
    by_video: dict[str, list[dict]] = defaultdict(list)
    for c in clips:
        by_video[c["video_id"]].append(c)
    out: dict[str, str] = {}
    for video_id, vclips in by_video.items():
        vclips = sorted(vclips, key=lambda c: c["idx"])
        durs = [_dur(c) for c in vclips]
        total = sum(durs)
        split = ["train"] * len(vclips)
        if total >= min_video_s and len(vclips) >= 2:
            rng = random.Random(f"{seed}:{video_id}")
            taken: set[int] = set()
            for name, ratio in (("val", ratios[1]), ("test", ratios[2])):
                target = ratio * total
                full, partial = [], []
                for s in range(len(vclips)):
                    n, complete = _run_len(durs, s, target)
                    span = set(range(s, s + n))
                    if s + n <= len(vclips) and not span & taken:
                        (full if complete else partial).append(span)
                options = full or partial
                if not options:
                    continue
                span = rng.choice(options)
                taken |= span
                for i in span:
                    split[i] = name
        for c, s in zip(vclips, split):
            out[c["id"]] = s
    return out
