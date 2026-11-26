import numpy as np
import pytest
import soundfile as sf

from ytds.config import Thresholds
from ytds.pipeline.quality import load_ecapa, quality, speaker_filter


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_speaker_filter_drops_outlier():
    base = np.zeros(8)
    base[0] = 1.0
    rng = np.random.default_rng(0)
    emb = [_unit(base + rng.normal(0, 0.05, 8)) for _ in range(9)]
    other = np.zeros(8)
    other[1] = 1.0
    emb.append(other)
    sims, keep = speaker_filter(np.array(emb), np.full(10, 3.5), 0.6)
    assert keep.tolist() == [True] * 9 + [False]
    assert sims[9] < 0.2 and sims[0] > 0.9


def test_speaker_filter_single_clip_kept():
    sims, keep = speaker_filter(np.ones((1, 4)), np.array([3.0]), 0.6)
    assert sims.tolist() == [1.0] and keep.tolist() == [True]


def test_quality_sets_drop_reasons(tmp_path):
    sr = 44100
    wav = tmp_path / "vocals.wav"
    sf.write(str(wav), np.zeros(sr * 40, dtype="float32"), sr, subtype="PCM_16")
    clips = [{"idx": i, "start_s": i * 4.0, "end_s": i * 4.0 + 3.0} for i in range(6)]
    scores = {0: 3.5, 1: 3.6, 2: 3.4, 3: 3.5, 4: 2.0, 5: 3.5}
    vecs = {0: [1, 0], 1: [1, 0.05], 2: [1, -0.05], 3: [1, 0.02], 4: [0, 1], 5: [0, 1]}
    # identify the clip by a marker: the fake audio is silent, so key off call order
    order = iter(range(6))
    ovrl = iter(range(6))

    def scorer(y):
        i = next(ovrl)
        return (3.0, 3.0, 3.0, scores[i])

    def embedder(y):
        return np.array(vecs[next(order)], dtype=float)

    out = quality(wav, clips, Thresholds(), scorer=scorer, embedder=embedder)
    assert [c["drop_reason"] for c in out] == [None, None, None, None, "dnsmos", "other_speaker"]
    assert [c.get("status") for c in out] == [None] * 4 + ["dropped", "dropped"]
    assert out[0]["dnsmos"] == 3.5
    assert out[5]["speaker_sim"] < 0.6 < out[0]["speaker_sim"]


@pytest.mark.gpu
def test_real_dnsmos_and_ecapa(tmp_path):
    sr = 44100
    t = np.arange(sr * 4) / sr
    y = (0.2 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype("float32")
    wav = tmp_path / "vocals.wav"
    sf.write(str(wav), y, sr, subtype="PCM_16")
    clips = [{"idx": 0, "start_s": 0.0, "end_s": 2.0}, {"idx": 1, "start_s": 2.0, "end_s": 4.0}]
    out = quality(wav, clips, Thresholds(), embedder=load_ecapa())
    print(out)
    assert all(isinstance(c["dnsmos"], float) for c in out)
    assert all(-1.0 <= c["speaker_sim"] <= 1.0001 for c in out)
