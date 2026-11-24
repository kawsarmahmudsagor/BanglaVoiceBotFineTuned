from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from test_select import make_wd

from dataprep.__main__ import main
from dataprep.steps import brouhaha as bh
from dataprep.steps import dnsmos as dm
from dataprep.workdir import StepFile, WorkDir


def _wav(path, sr=16000):
    sf.write(path, (0.3 * np.sin(np.arange(sr) / 10)).astype("float32"), sr)


def _clips(wd: WorkDir, paths: dict[str, object]) -> None:
    StepFile.write(
        wd.clips_path,
        pd.DataFrame({"id": list(paths), "src_path": [str(p) for p in paths.values()], "text_raw": "x", "src_split": "train"}),
        step="import", version=1, params={},
    )


def test_score_dnsmos_maps_scorer_output():
    out = dm.score_dnsmos(np.zeros(16000, dtype="float32"), scorer=lambda y: (3.1, 3.2, 3.3, 3.4))
    assert out == {"dnsmos_p808": 3.1, "dnsmos_sig": 3.2, "dnsmos_bak": 3.3, "dnsmos_ovrl": 3.4}


def test_dnsmos_step_resume_and_errors(tmp_path, monkeypatch):
    good, bad = tmp_path / "g.wav", tmp_path / "b.wav"
    _wav(good)
    bad.write_bytes(b"not audio")
    wd = WorkDir(tmp_path / "work")
    _clips(wd, {"good": good, "bad": bad})
    calls: list[int] = []

    def fake(y):
        calls.append(len(y))
        return (3.0, 3.1, 3.2, 3.3)

    monkeypatch.setattr(dm, "_scorer", fake)
    args = ["dnsmos", "--work", str(wd.root)]
    assert main(args) == 0
    df = pd.read_parquet(wd.step_path("dnsmos")).set_index("id")
    assert list(df.columns) == ["dnsmos_p808", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "error"]
    assert df.loc["good", "dnsmos_ovrl"] == pytest.approx(3.3) and pd.isna(df.loc["good", "error"])
    assert df.loc["bad", "error"]
    assert calls == [16000]
    assert main(args) == 0  # resume: nothing recomputed, errored row kept
    assert calls == [16000]
    assert len(pd.read_parquet(wd.step_path("dnsmos"))) == 2


def _frames(spans, n=100, snr=20.0, c50=30.0):
    f = np.zeros((n, 3))
    for a, b in spans:
        f[a:b, 0] = 1.0
    f[:, 1], f[:, 2] = snr, c50
    return f


def test_max_pause_from_segments():
    assert bh.max_pause([[0, 1], [2.5, 4]]) == pytest.approx(1.5)
    assert bh.max_pause([[0, 1], [1.2, 2], [4, 5]]) == pytest.approx(2.0)
    assert bh.max_pause([[0, 1]]) == 0.0


def test_score_brouhaha_derives_values():
    # 10 ms frames: speech 0-1 s and 2.5-4 s
    frames = _frames([(0, 100), (250, 400)], n=500)
    frames[:100, 1] = 10.0
    frames[250:400, 1] = 20.0
    out = bh.score_brouhaha("x.wav", scorer=lambda p: (0.01, frames))
    assert out["speech_segments"] == [[0.0, 1.0], [2.5, 4.0]]
    assert out["max_pause_s"] == pytest.approx(1.5)
    assert out["snr_db"] == pytest.approx((100 * 10 + 150 * 20) / 250)  # mean over speech frames only
    assert out["c50_db"] == pytest.approx(30.0)


def test_score_brouhaha_no_speech_raises():
    with pytest.raises(ValueError, match="no speech"):
        bh.score_brouhaha("x.wav", scorer=lambda p: (0.01, _frames([])))


def test_brouhaha_step(tmp_path, monkeypatch):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    _wav(a)
    _wav(b)
    wd = WorkDir(tmp_path / "work")
    _clips(wd, {"a": a, "b": b})
    seen: list[str] = []

    def fake(path):
        seen.append(path.name)
        return (0.01, _frames([(0, 100), (250, 400)], n=500) if path.name == "a.wav" else _frames([]))

    monkeypatch.setattr(bh, "_scorer", fake)
    args = ["brouhaha", "--work", str(wd.root)]
    assert main(args) == 0
    df = pd.read_parquet(wd.step_path("brouhaha")).set_index("id")
    assert df.loc["a", "max_pause_s"] == pytest.approx(1.5)
    assert [list(s) for s in df.loc["a", "speech_segments"]] == [[0.0, 1.0], [2.5, 4.0]]
    assert df.loc["b", "error"] == "ValueError: no speech"
    assert main(args) == 0
    assert seen == ["a.wav", "b.wav"]  # second run recomputed nothing
    assert main([*args, "--retry-errors"]) == 0
    assert seen == ["a.wav", "b.wav", "b.wav"]


def test_brouhaha_only_profile(tmp_path, monkeypatch):
    wd = make_wd(tmp_path / "work", overrides={1: {"clip_frac": 0.5}, 2: {"clip_frac": 0.5}}, n=5)
    prof = tmp_path / "p.yaml"
    prof.write_text(
        "name: p\ntask: tts\nfilters:\n"
        "  - {kind: range, column: clip_frac, max: 0.1}\n"
        "  - {kind: range, column: snr_db, min: 10}\n",
        encoding="utf-8",
    )

    def fake(path):
        return (0.01, _frames([(0, 100)]))

    monkeypatch.setattr(bh, "_scorer", fake)
    assert main(["brouhaha", "--work", str(wd.root), "--only-profile", str(prof)]) == 0
    df = pd.read_parquet(wd.step_path("brouhaha"))
    assert sorted(df["id"]) == ["r0", "r3", "r4"]  # r1, r2 fail the clip_frac filter; the snr filter is ignored


def test_brouhaha_only_profile_shipped_tts_loose(tmp_path, monkeypatch, capsys):
    # Only text + audio_stats exist (spec 5.2 step 2): no speakers, no cer, no dnsmos, no brouhaha.
    wd = make_wd(tmp_path / "work", overrides={1: {"dur": 30.0}}, n=5)  # r1 fails tts-loose dur <= 15
    monkeypatch.setattr(bh, "_scorer", lambda path: (0.01, _frames([(0, 100)])))
    assert main(["brouhaha", "--work", str(wd.root), "--only-profile", "tts-loose"]) == 0
    assert "CER filter skipped" in capsys.readouterr().out
    df = pd.read_parquet(wd.step_path("brouhaha"))
    assert sorted(df["id"]) == ["r0", "r2", "r3", "r4"]


def test_brouhaha_fails_fast_when_scorer_cannot_load(tmp_path, monkeypatch, capsys):
    wd = make_wd(tmp_path / "work", n=3)
    monkeypatch.setattr(bh, "_scorer", None)

    def boom():
        raise RuntimeError("HF_TOKEN is not set")

    monkeypatch.setattr(bh, "make_scorer", boom)
    assert main(["brouhaha", "--work", str(wd.root)]) == 1
    assert "HF_TOKEN is not set" in capsys.readouterr().out
    assert not wd.step_path("brouhaha").exists()


def test_dnsmos_fails_fast_when_scorer_cannot_load(tmp_path, monkeypatch, capsys):
    wd = make_wd(tmp_path / "work", n=3)
    monkeypatch.setattr(dm, "_scorer", None)

    def boom():
        raise ImportError("No module named 'torchmetrics'")

    monkeypatch.setattr(dm, "make_scorer", boom)
    assert main(["dnsmos", "--work", str(wd.root)]) == 1
    assert "torchmetrics" in capsys.readouterr().out
    assert not wd.step_path("dnsmos").exists()
