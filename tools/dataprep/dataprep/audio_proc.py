"""TTS export audio processing (spec 8.3): mono, trim to speech, loudness, true-peak limit, resample."""
from __future__ import annotations

import numpy as np

OVERSAMPLE = 4
PAD_S = 0.15
MIN_SPAN_S = 0.4  # pyloudnorm's gating block; shorter trims cannot be loudness-normalised


def trim_bounds_s(segments, dur: float, pad_s: float = PAD_S) -> tuple[float, float] | None:
    """(start, end) in seconds that `process_tts` keeps: first/last segment +- pad, clamped to [0, dur].

    None when there are no usable segments. This is the one place the trim formula lives; `select`
    (speech_span_s) and the export pre-flight use it too.
    """
    if segments is None or isinstance(segments, float) or len(segments) == 0:
        return None
    seg = np.asarray([np.asarray(s, dtype="float64") for s in segments]).reshape(-1, 2)
    return max(0.0, float(seg[:, 0].min()) - pad_s), min(float(dur), float(seg[:, 1].max()) + pad_s)


def speech_span_s(segments, dur: float, pad_s: float = PAD_S) -> float:
    """Length of the trimmed clip in seconds; NaN without segments."""
    b = trim_bounds_s(segments, dur, pad_s)
    return float("nan") if b is None else b[1] - b[0]


def true_peak_db(y: np.ndarray, sr: int) -> float:
    """True-peak estimate in dBTP: sample peak of the signal oversampled 4x with soxr."""
    import soxr

    peak = float(np.max(np.abs(soxr.resample(y, sr, sr * OVERSAMPLE, quality="HQ"))))
    return -np.inf if peak <= 0 else 20.0 * np.log10(peak)


def _limit_true_peak(y: np.ndarray, sr: int, ceiling_db: float) -> np.ndarray:
    tp = true_peak_db(y, sr)
    if tp > ceiling_db:
        y = (y * np.float32(10.0 ** ((ceiling_db - tp) / 20.0))).astype("float32")
    return y


def process_tts(
    y: np.ndarray, sr: int, segments, *, pad_s: float = 0.15, target_lufs: float = -23.0,
    peak_dbtp: float = -1.0, out_sr: int = 24000,
) -> np.ndarray:
    """Return float32 mono audio at `out_sr`, in spec 8.3 order.

    Trim keeps `pad_s` around the first and last speech segment (clamped to the clip). Loudness is normalised to
    `target_lufs` integrated; if the true peak then exceeds `peak_dbtp`, a plain gain reduction brings it to the
    ceiling, so such clips end slightly BELOW target_lufs. Resampling can overshoot a little, so the ceiling is
    checked once more at `out_sr`. Raises ValueError for no segments, empty trim, or silence.
    """
    import pyloudnorm
    import soxr

    y = np.asarray(y, dtype="float32")
    if y.ndim > 1:  # (samples, channels) as soundfile gives, or (channels, samples) as librosa gives
        y = y.mean(axis=1) if y.shape[1] <= y.shape[0] else y.mean(axis=0)
    bounds = trim_bounds_s(segments, len(y) / sr, pad_s)
    if bounds is None:
        raise ValueError("no speech segments to trim to")
    start, end = int(round(bounds[0] * sr)), min(len(y), int(round(bounds[1] * sr)))
    if end <= start:
        raise ValueError("speech segments lie outside the audio")
    if (end - start) / sr < MIN_SPAN_S:
        raise ValueError(f"trimmed clip is {(end - start) / sr:.2f}s, under {MIN_SPAN_S}s; cannot normalise loudness")
    y = y[start:end]

    loudness = pyloudnorm.Meter(sr).integrated_loudness(y.astype("float64"))
    if not np.isfinite(loudness):
        raise ValueError("clip is silent; cannot normalise loudness")
    y = (y * np.float32(10.0 ** ((target_lufs - loudness) / 20.0))).astype("float32")
    y = _limit_true_peak(y, sr, peak_dbtp)
    if sr != out_sr:
        y = soxr.resample(y, sr, out_sr).astype("float32")
        y = _limit_true_peak(y, out_sr, peak_dbtp)
    return y
