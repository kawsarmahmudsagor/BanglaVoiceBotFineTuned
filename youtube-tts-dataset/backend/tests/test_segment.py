import random

import numpy as np
import pytest

from ytds.config import Thresholds
from ytds.pipeline import segment as seg
from ytds.pipeline.segment import plan_clips

KW = dict(merge_gap_s=0.3, min_s=2.0, max_s=15.0)


def test_merges_short_gaps():
    assert plan_clips([(0.0, 1.5), (1.7, 3.5)], **KW) == [(0.0, 3.5)]


def test_keeps_long_gaps_apart():
    assert plan_clips([(0.0, 2.5), (3.5, 6.0)], **KW) == [(0.0, 2.5), (3.5, 6.0)]


def test_splits_long_span_at_longest_gap():
    regions = [(0.0, 5.0), (5.1, 9.0), (9.25, 12.0), (12.1, 18.0)]
    # gaps: 0.1, 0.25, 0.1 all merge -> span 0-18 > 15 -> cut at the 0.25 gap
    assert plan_clips(regions, **KW) == [(0.0, 9.0), (9.25, 18.0)]


def test_hard_cut_without_gap():
    assert plan_clips([(0.0, 31.0)], **KW) == [(0.0, 15.0), (15.0, 30.0)]


def test_short_tail_merges_with_neighbour():
    out = plan_clips([(0.0, 5.0), (5.5, 6.0), (20.0, 25.0)], **KW)
    assert out == [(0.0, 6.0), (20.0, 25.0)]


def test_isolated_short_dropped():
    assert plan_clips([(0.0, 0.8), (20.0, 24.0)], **KW) == [(20.0, 24.0)]


def test_short_not_merged_when_too_long():
    assert plan_clips([(0.0, 14.0), (14.5, 15.5)], **KW) == [(0.0, 14.0)]


def test_no_regions_gives_no_clips():
    assert plan_clips([], **KW) == []


def test_all_clips_within_limits():
    rng = random.Random(7)
    for _ in range(200):
        t, regions = 0.0, []
        for _ in range(rng.randint(0, 40)):
            t += rng.uniform(0.0, 2.0)
            d = rng.uniform(0.1, 20.0)
            regions.append((t, t + d))
            t += d
        for s, e in plan_clips(regions, **KW):
            assert 2.0 - 1e-9 <= e - s <= 15.0 + 1e-9


def test_segment_builds_clip_dicts(tmp_path, monkeypatch):
    import soundfile as sf

    wav = tmp_path / "vocals.wav"
    sf.write(str(wav), np.zeros(44100 * 12, dtype="float32"), 44100, subtype="PCM_16")
    monkeypatch.setattr(seg, "vad_regions", lambda y: [(0.5, 3.0), (6.0, 9.0)])
    clips = seg.segment(wav, Thresholds())
    assert clips == [
        {"idx": 0, "start_s": 0.5, "end_s": 3.0},
        {"idx": 1, "start_s": 6.0, "end_s": 9.0},
    ]
