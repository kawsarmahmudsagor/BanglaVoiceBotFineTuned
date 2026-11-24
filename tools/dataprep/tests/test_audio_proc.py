from __future__ import annotations

import numpy as np
import pyloudnorm
import pytest

from dataprep.audio_proc import process_tts, true_peak_db


def tone(sr, secs, amp=0.05, f=300.0):
    t = np.arange(int(sr * secs)) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype("float32")


def lufs(y, sr):
    return pyloudnorm.Meter(sr).integrated_loudness(y.astype("float64"))


def test_output_rate_and_loudness():
    sr = 22050
    out = process_tts(tone(sr, 4.0), sr, [[0.5, 3.5]])
    assert out.dtype == np.float32 and out.ndim == 1
    assert abs(lufs(out, 24000) + 23.0) <= 1.0
    assert true_peak_db(out, 24000) <= -1.0


def test_trim_keeps_150ms_padding():
    sr = 24000
    out = process_tts(tone(sr, 5.0), sr, [[1.0, 2.0], [3.0, 3.5]])
    assert abs(len(out) / sr - (3.5 + 0.15 - (1.0 - 0.15))) < 0.01


def test_trim_clamps_to_clip():
    sr = 24000
    out = process_tts(tone(sr, 2.0), sr, [[0.05, 1.95]])
    assert abs(len(out) / sr - 2.0) < 0.01


def test_true_peak_limited():
    sr = 24000
    y = np.random.default_rng(0).normal(0, 0.02, sr * 3).astype("float32")
    y[sr] = 1.0  # a click far above the loudness-normalised body
    out = process_tts(y, sr, [[0.0, 3.0]])
    assert true_peak_db(out, 24000) <= -1.0 + 1e-3


def test_stereo_to_mono_and_other_rates():
    sr = 16000
    y = np.stack([tone(sr, 3.0), tone(sr, 3.0)], axis=1)
    out = process_tts(y, sr, [[0.0, 3.0]])
    assert out.ndim == 1 and abs(len(out) / 24000 - 3.0) < 0.01


def test_no_segments_raises():
    with pytest.raises(ValueError, match="segments"):
        process_tts(tone(24000, 2.0), 24000, [])


def test_silence_raises():
    with pytest.raises(ValueError):
        process_tts(np.zeros(48000, dtype="float32"), 24000, [[0.0, 2.0]])
