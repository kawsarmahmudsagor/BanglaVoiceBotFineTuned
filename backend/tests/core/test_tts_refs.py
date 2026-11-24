from app.core.tts_refs import pick_reference


def row(sid, speaker, dur):
    return {"sample_id": sid, "speaker": speaker, "duration_s": dur}


def test_pick_reference_median_per_speaker():
    rows = [row("a1", "x", 1.0), row("a2", "x", 5.0), row("a3", "x", 3.0),
            row("b1", "y", 2.0), row("b2", "y", 4.0),  # even count: lower-middle
            row("c1", "z", 2.0), row("c0", "z", 2.0), row("c2", "z", 2.0)]  # ties by sample_id
    got = pick_reference(rows)
    assert got["x"]["sample_id"] == "a3"
    assert got["y"]["sample_id"] == "b1"
    assert got["z"]["sample_id"] == "c1"


def test_pick_reference_default_speaker():
    got = pick_reference([{"sample_id": "s", "duration_s": 1.0}])
    assert list(got) == ["default"]
