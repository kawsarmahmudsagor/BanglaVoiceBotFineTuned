import numpy as np
import pytest
import soundfile as sf
import torch

from ytds.pipeline.separate import load_demucs, separate


class FakeDemucs(torch.nn.Module):
    """Stands in for htdemucs: 4 stems; stem k is the constant (k+1)*0.1."""

    sources = ["drums", "bass", "other", "vocals"]
    samplerate = 44100
    audio_channels = 2
    segment = 7.0
    num_parameters = 0

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, t = x.shape
        out = torch.zeros(b, 4, c, t)
        for k in range(4):
            out[:, k] = (k + 1) * 0.1
        return out


def test_separate_writes_mono_vocals(tmp_path):
    src = tmp_path / "audio.wav"
    sig = (0.1 * np.sin(2 * np.pi * 100 * np.arange(48000 * 130) / 48000)).astype(np.float32)
    sf.write(src, sig, 48000, subtype="FLOAT")
    out = tmp_path / "vocals.wav"
    prog = []
    separate(src, out, FakeDemucs(), 7.0, lambda d, t: prog.append((d, t)))
    data, sr = sf.read(out, dtype="float32", always_2d=True)
    info = sf.info(out)
    assert sr == 44100 and info.channels == 1 and info.subtype == "PCM_16"
    assert abs(len(data) - 130 * 44100) <= 2
    # vocals is index 3 -> 0.4 in normalized units, de-normalized: 0.4*std+mean
    mid = data[44100 * 30 : 44100 * 31, 0]
    assert np.allclose(mid, 0.4 * 0.1 / np.sqrt(2), atol=1e-3)
    assert prog == [(1, 3), (2, 3), (3, 3)]


@pytest.mark.gpu
def test_real_htdemucs_on_synthetic(tmp_path):
    assert torch.cuda.is_available()
    t = np.arange(48000 * 10) / 48000
    sig = (0.3 * np.sin(2 * np.pi * 220 * t) * (1 + np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    src = tmp_path / "audio.wav"
    sf.write(src, sig, 48000, subtype="PCM_16")
    model = load_demucs()
    out = tmp_path / "vocals.wav"
    prog = []
    separate(src, out, model, 7.0, lambda d, t: prog.append((d, t)))
    info = sf.info(out)
    assert info.samplerate == 44100 and info.channels == 1
    assert abs(info.frames - 10 * 44100) <= 2
    assert prog == [(1, 1)]


class PointwiseDemucs(FakeDemucs):
    """Input-dependent but pointwise: vocals = tanh(x), other stems zero."""

    def forward(self, x):
        b, c, t = x.shape
        out = torch.zeros(b, 4, c, t)
        out[:, 3] = torch.tanh(x)
        return out


def test_blocks_match_unblocked_and_length_preserved(tmp_path):
    rng = np.random.default_rng(0)
    n = 48000 * 130
    sig = (0.2 * rng.standard_normal(n) + 0.05).astype(np.float32).clip(-1, 1)
    src = tmp_path / "audio.wav"
    sf.write(src, sig, 48000, subtype="FLOAT")
    out = tmp_path / "v.wav"
    separate(src, out, PointwiseDemucs(), 7.0, lambda d, t: None)
    got, sr = sf.read(out, dtype="float32")
    from scipy.signal import resample_poly

    ref = resample_poly(sig, 147, 160).astype(np.float32)
    assert len(got) == len(ref)
    expect = np.tanh((ref - ref.mean()) / ref.std()) * ref.std() + ref.mean()
    assert np.abs(got - expect).max() < 2e-3
    for b in (60 * 44100, 120 * 44100):
        assert np.abs(got[b - 50 : b + 50] - expect[b - 50 : b + 50]).max() < 2e-3


def test_silent_file_writes_silence(tmp_path):
    src = tmp_path / "audio.wav"
    sf.write(src, np.zeros(48000 * 3, dtype=np.float32), 48000, subtype="PCM_16")
    out = tmp_path / "v.wav"
    separate(src, out, FakeDemucs(), 7.0, lambda d, t: None)
    got, _ = sf.read(out, dtype="float32")
    assert len(got) > 0 and np.isfinite(got).all() and np.abs(got).max() == 0
