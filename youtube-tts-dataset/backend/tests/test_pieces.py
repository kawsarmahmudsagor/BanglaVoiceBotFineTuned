import threading
import time

import numpy as np

from ytds.config import Thresholds
from ytds.pipeline import pieces as pc
from ytds.pipeline.pieces import plan_pieces

SR = 16000


def _noise(seconds, silent_at=()):
    rng = np.random.default_rng(0)
    y = (rng.standard_normal(int(seconds * SR)) * 0.1).astype("float32")
    for t in silent_at:
        a = int(t * SR) - 160
        y[a:a + 320] = 0.0
    return y


def _se(ps):
    return [(p["start_s"], p["end_s"]) for p in ps]


def test_merges_short_gaps_and_splits_long_ones():
    y = _noise(20)
    ps = plan_pieces([(0, 3), (3.1, 6), (7, 9)], [(0, 0.0, 20.0)], y, 10.0, 0.3)
    assert _se(ps) == [(0, 6), (7, 9)]
    assert [p["idx"] for p in ps] == [0, 1] and {p["range_idx"] for p in ps} == {0}


def test_long_region_split_at_quietest_frames():
    y = _noise(24, silent_at=(9.2, 18.6))
    ps = plan_pieces([(0.0, 23.0)], [(0, 0.0, 24.0)], y, 10.0, 0.3)
    assert len(ps) == 3
    assert abs(ps[0]["end_s"] - 9.2) <= 0.02 and abs(ps[1]["end_s"] - 18.6) <= 0.02
    assert ps[1]["start_s"] == ps[0]["end_s"] and ps[2]["start_s"] == ps[1]["end_s"]
    assert all(p["end_s"] - p["start_s"] <= 10.0 + 1e-9 for p in ps)
    assert ps[-1]["end_s"] == 23.0


def test_region_crossing_span_boundary_is_clipped():
    y = _noise(30)
    ps = plan_pieces([(8.0, 14.0)], [(0, 0.0, 10.0), (1, 10.0, 30.0)], y, 10.0, 0.3)
    assert _se(ps) == [(8.0, 10.0), (10.0, 14.0)]
    assert [p["range_idx"] for p in ps] == [0, 1] and [p["idx"] for p in ps] == [0, 1]


def test_tiny_piece_merges_across_gap_when_it_fits():
    y = _noise(20)
    ps = plan_pieces([(0, 0.5), (1.5, 4)], [(0, 0.0, 20.0)], y, 10.0, 0.3)
    assert _se(ps) == [(0, 4)]


def test_no_regions():
    assert plan_pieces([], [(0, 0.0, 5.0)], _noise(5), 10.0, 0.3) == []


def test_vad_regions_serialised(monkeypatch):
    import silero_vad

    state = {"active": 0, "overlap": False}

    def fake_ts(wav, model, **kw):
        state["active"] += 1
        if state["active"] > 1:
            state["overlap"] = True
        time.sleep(0.05)
        state["active"] -= 1
        return [{"start": 0.0, "end": 1.0}]

    monkeypatch.setattr(silero_vad, "get_speech_timestamps", fake_ts)
    monkeypatch.setattr(pc, "_get_vad", lambda: object())
    y = np.zeros(16000, dtype="float32")
    results = []
    ts = [threading.Thread(target=lambda: results.append(pc.vad_regions(y))) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert results == [[(0.0, 1.0)]] * 4
    assert not state["overlap"]


def test_pieces_wires_vad_and_spans(tmp_path, monkeypatch):
    wav = tmp_path / "v.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(pc, "load_16k", lambda p: _noise(20))
    monkeypatch.setattr(pc, "vad_regions", lambda y: [(0.5, 3.0), (6.0, 9.0)])
    cmap = [{"range_idx": 0, "cut_start_s": 0.0, "cut_end_s": 20.0}]
    ps = pc.pieces(wav, cmap, Thresholds())
    assert _se(ps) == [(0.5, 3.0), (6.0, 9.0)]
