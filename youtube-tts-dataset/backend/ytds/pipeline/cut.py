"""Cut stage: join user-chosen keep ranges into one file with a time map."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

GAP_S = 2.0
MIN_RANGE_S = 1.0


class BadCuts(ValueError):
    pass


def normalize_ranges(cuts, duration_s: float) -> list[tuple[float, float]]:
    if not cuts:
        return [(0.0, float(duration_s))]
    rs = []
    for c in cuts:
        s, e = float(c[0]), float(c[1])
        if s >= e:
            raise BadCuts(f"range start {s} must be before end {e}")
        s, e = max(0.0, s), min(float(duration_s), e)
        if s < e:
            rs.append((s, e))
    rs.sort()
    merged: list[list[float]] = []
    for s, e in rs:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out = [(s, e) for s, e in merged if e - s >= MIN_RANGE_S]
    if not out:
        raise BadCuts("no usable ranges")
    return out


def cut_audio(audio_wav: Path, out_dir: Path, cuts) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(str(audio_wav)) as f:
        sr, ch = f.samplerate, f.channels
        ranges = normalize_ranges(cuts, f.frames / sr)
        chunks, cmap, pos = [], [], 0.0
        gap = np.zeros((int(round(GAP_S * sr)), ch), dtype="float32")
        for i, (s, e) in enumerate(ranges):
            f.seek(int(round(s * sr)))
            a = f.read(int(round((e - s) * sr)), dtype="float32", always_2d=True)
            if i:
                chunks.append(gap)
                pos += GAP_S
            dur = len(a) / sr
            cmap.append({"range_idx": i, "cut_start_s": pos, "cut_end_s": pos + dur, "orig_start_s": s})
            chunks.append(a)
            pos += dur
    data = np.concatenate(chunks)
    sf.write(str(out_dir / "audio.wav"), data if ch > 1 else data[:, 0], sr, subtype="PCM_16")
    (out_dir / "map.json").write_text(json.dumps(cmap), encoding="utf-8")
    return cmap


def to_orig(t: float, cmap: list[dict]) -> float:
    prev = None
    for m in cmap:
        if t < m["cut_start_s"]:
            break
        prev = m
        if t <= m["cut_end_s"]:
            return m["orig_start_s"] + (t - m["cut_start_s"])
    if prev is None:
        prev = cmap[0]
        return prev["orig_start_s"]
    return prev["orig_start_s"] + (prev["cut_end_s"] - prev["cut_start_s"])


def range_spans(cmap) -> list[tuple[int, float, float]]:
    return [(m["range_idx"], m["cut_start_s"], m["cut_end_s"]) for m in cmap]
