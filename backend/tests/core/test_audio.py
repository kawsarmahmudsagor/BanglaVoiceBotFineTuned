import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
pytest.importorskip("soxr")

from app.core.backend_kit.audio import duration_s, load_audio, write_wav  # noqa: E402


def test_load_audio_resamples_and_downmixes(tmp_path):
    p = tmp_path / "a.wav"
    sr = 44100
    t = np.arange(sr) / sr
    stereo = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 220 * t)], axis=1).astype("float32")
    sf.write(p, stereo, sr)
    a = load_audio(p, 16000)
    assert a.dtype == np.float32 and a.ndim == 1
    assert abs(len(a) - 16000) <= 2
    assert duration_s(p) == pytest.approx(1.0, abs=0.01)


def test_write_wav_roundtrip(tmp_path):
    p = tmp_path / "sub" / "b.wav"
    x = (0.5 * np.sin(np.linspace(0, 100, 16000))).astype("float32")
    write_wav(p, x, 16000)
    assert sf.info(p).subtype == "PCM_16"
    y = load_audio(p, 16000)
    assert len(y) == 16000 and np.allclose(x, y, atol=1e-3)
