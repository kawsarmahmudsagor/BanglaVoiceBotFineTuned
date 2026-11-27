import json

import numpy as np
import pytest
import soundfile as sf

from ytds.pipeline.cut import BadCuts, cut_audio, normalize_ranges, range_spans, to_orig


def test_bad_order():
    with pytest.raises(BadCuts):
        normalize_ranges([[5, 3]], 60)


def test_merge_and_drop():
    assert normalize_ranges([[10, 20], [15, 30], [50, 50.5], [70, 80]], 60) == [(10, 30)]


def test_none_whole():
    assert normalize_ranges(None, 60) == [(0.0, 60.0)]
    assert normalize_ranges([], 60) == [(0.0, 60.0)]


def test_all_beyond():
    with pytest.raises(BadCuts):
        normalize_ranges([[70, 80]], 60)


def test_cut_audio(tmp_path):
    sr = 48000
    t = np.arange(sr * 10) / sr
    src = tmp_path / "in.wav"
    sf.write(src, (0.5 * np.sin(2 * np.pi * 440 * t)).astype("float32"), sr, subtype="PCM_16")
    out = tmp_path / "out"
    cmap = cut_audio(src, out, [[6, 8], [1, 3]])
    data, osr = sf.read(out / "audio.wav")
    assert osr == sr
    assert len(data) == 6 * sr
    assert not data[2 * sr : 4 * sr].any()
    assert [m["cut_start_s"] for m in cmap] == [0.0, 4.0]
    assert json.loads((out / "map.json").read_text()) == cmap
    assert to_orig(5.0, cmap) == pytest.approx(7.0)
    assert to_orig(3.0, cmap) == pytest.approx(3.0)
    assert to_orig(99, cmap) == pytest.approx(8.0)
    assert range_spans(cmap) == [(0, 0.0, 2.0), (1, 4.0, 6.0)]


def test_cut_none_whole(tmp_path):
    src = tmp_path / "in.wav"
    sf.write(src, np.zeros((16000 * 3, 2), dtype="float32"), 16000, subtype="PCM_16")
    cmap = cut_audio(src, tmp_path / "o", None)
    assert cmap == [{"range_idx": 0, "cut_start_s": 0.0, "cut_end_s": 3.0, "orig_start_s": 0.0}]
    assert sf.read(tmp_path / "o" / "audio.wav")[0].shape == (48000, 2)
