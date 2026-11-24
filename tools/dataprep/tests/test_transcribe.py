from __future__ import annotations

import pandas as pd
import pytest
from test_select import make_wd, prof

from dataprep.__main__ import main
from dataprep.profiles import Range
from dataprep.select import cer_step_name, select
from dataprep.steps import transcribe as tr
from dataprep.workdir import StepFile

REF = "কখগঘ"


class Fake:
    """Labeller returning a fixed hypothesis per path stem; records batches; raises for batches containing `bad`."""

    def __init__(self, hyps: dict[str, str], bad: frozenset[str] = frozenset()):
        self.hyps, self.bad, self.batches = hyps, bad, []

    def __call__(self, paths):
        self.batches.append([p.stem for p in paths])
        if any(p.stem in self.bad for p in paths):
            raise RuntimeError("bad file")
        return [self.hyps[p.stem] for p in paths]


def _wd(tmp_path, n=4, extra=None):
    extra = extra or {}
    wd = make_wd(tmp_path, n=n, overrides={i: {"text": REF, **extra.get(i, {})} for i in range(n)})
    clips = pd.read_parquet(wd.clips_path)
    clips["src_path"] = [str(tmp_path / f"r{i}.wav") for i in range(n)]
    StepFile.write(wd.clips_path, clips, step="import", version=1, params={})
    return wd


def _read(wd, spec):
    return pd.read_parquet(wd.step_path(cer_step_name(spec))).set_index("id")


def test_slug_name_is_the_select_name():
    assert tr.step_name("nemo:bengaliAI/BanglaConformer") == cer_step_name("nemo:bengaliAI/BanglaConformer")
    assert tr.step_name("whisper-hf:/m/x") == "cer_whisper-hf--m-x"


def test_cer_values(tmp_path, monkeypatch):
    wd = _wd(tmp_path, n=3)
    monkeypatch.setattr(tr, "_labeller", Fake({"r0": REF, "r1": "কখগচ", "r2": ""}))
    assert tr.run(wd, "fake:1") == 3
    df = _read(wd, "fake:1")
    assert list(df.columns) == ["hyp", "cer", "error"]
    assert df.loc["r0", "cer"] == 0.0 and df.loc["r1", "cer"] == pytest.approx(0.25) and df.loc["r2", "cer"] == pytest.approx(1.0)
    assert df.loc["r1", "hyp"] == "কখগচ" and df["error"].isna().all()


def test_cer_normalizes_both_sides(tmp_path, monkeypatch):
    wd = _wd(tmp_path, n=1)
    monkeypatch.setattr(tr, "_labeller", Fake({"r0": "ক খ গ ঘ।"}))  # spacing and danda are normalized away
    tr.run(wd, "fake:1")
    assert _read(wd, "fake:1").loc["r0", "cer"] == 0.0


def test_cer_skips_empty_reference(tmp_path, monkeypatch):
    wd = _wd(tmp_path, n=2, extra={1: {"empty_asr": True}})
    fake = Fake({"r0": REF, "r1": "x"})
    monkeypatch.setattr(tr, "_labeller", fake)
    tr.run(wd, "fake:1")
    df = _read(wd, "fake:1")
    assert df.loc["r1", "error"] == "empty reference" and pd.isna(df.loc["r1", "cer"])
    assert fake.batches == [["r0"]]


def test_batching_and_batch_failure_falls_back_per_row(tmp_path, monkeypatch):
    wd = _wd(tmp_path, n=5)
    fake = Fake({f"r{i}": REF for i in range(5)}, bad=frozenset({"r1"}))
    monkeypatch.setattr(tr, "_labeller", fake)
    tr.run(wd, "fake:1", batch_size=2)
    df = _read(wd, "fake:1")
    assert df.loc["r1", "error"].startswith("RuntimeError") and pd.isna(df.loc["r1", "cer"])
    assert df.drop("r1")["error"].isna().all()
    assert fake.batches[0] == ["r0", "r1"] and ["r1"] in fake.batches  # batch first, then per-row retry
    assert all(len(b) <= 2 for b in fake.batches)


def test_resume_and_retry_errors(tmp_path, monkeypatch):
    wd = _wd(tmp_path, n=3)
    fake = Fake({f"r{i}": REF for i in range(3)}, bad=frozenset({"r2"}))
    monkeypatch.setattr(tr, "_labeller", fake)
    assert tr.run(wd, "fake:1") == 3
    fake.batches.clear()
    assert tr.run(wd, "fake:1") == 0 and fake.batches == []  # errored row kept, not retried
    fake.bad = frozenset()
    assert tr.run(wd, "fake:1", retry_errors=True) == 1 and fake.batches == [["r2"]]
    assert _read(wd, "fake:1")["error"].isna().all()


def test_select_reads_cer(tmp_path, monkeypatch):
    wd = _wd(tmp_path, n=2)
    monkeypatch.setattr(tr, "_labeller", Fake({"r0": REF, "r1": "কখচচ"}))
    tr.run(wd, "fake:1")
    res = select(wd, prof(filters=[Range(column="cer", max=0.3)]), "fake:1")
    assert set(res.rows["id"]) == {"r0"}


def test_cli_batches(tmp_path, monkeypatch):
    wd = _wd(tmp_path, n=3)
    fake = Fake({f"r{i}": REF for i in range(3)})
    monkeypatch.setattr(tr, "_labeller", fake)
    assert main(["transcribe", "--work", str(wd.root), "--labeller", "fake:1", "--batch-size", "8"]) == 0
    assert fake.batches == [["r0", "r1", "r2"]]


def test_only_profile_restricts_rows(tmp_path, monkeypatch):
    wd = _wd(tmp_path, n=3, extra={2: {"dur": 0.1}})  # r2 fails the profile's duration filter
    fake = Fake({f"r{i}": REF for i in range(3)})
    monkeypatch.setattr(tr, "_labeller", fake)
    monkeypatch.setattr("dataprep.profiles.load_profile", lambda name: prof(filters=[Range(column="dur", min=1.0), Range(column="cer", max=0.3)]))
    assert main(["transcribe", "--work", str(wd.root), "--labeller", "fake:1", "--only-profile", "t"]) == 0
    assert set(_read(wd, "fake:1").index) == {"r0", "r1"}


def test_fails_fast_when_labeller_cannot_load(tmp_path, monkeypatch, capsys):
    wd = _wd(tmp_path, n=2)
    monkeypatch.setattr(tr, "_labeller", None)

    def boom(spec, **kw):
        raise ImportError("No module named 'nemo'")

    monkeypatch.setattr(tr, "make_labeller", boom)
    assert main(["transcribe", "--work", str(wd.root), "--labeller", "nemo:x"]) == 1
    assert "nemo" in capsys.readouterr().out
    assert not wd.step_path(cer_step_name("nemo:x")).exists()


def test_unknown_labeller_scheme():
    with pytest.raises(ValueError, match="unknown labeller"):
        tr.make_labeller("foo:bar")


def test_nemo_return_shapes():
    class H:
        def __init__(self, t):
            self.text = t

    assert tr._texts(["a", "b"]) == ["a", "b"]
    assert tr._texts([H("a")]) == ["a"]
    assert tr._texts(([H("a")], [H("b")])) == ["a"]  # (best, all) tuple from RNNT models
