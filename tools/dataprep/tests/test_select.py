from __future__ import annotations

import json

import pandas as pd
import pytest

from dataprep.profiles import Flag, Percentile, Profile, Range, load_profile
from dataprep.select import SelectError, select
from dataprep.workdir import StaleStepError, StepFile, WorkDir

BASE = dict(text="আমি ভাত খাই", split="train", dur=5.0, clip_frac=0.0, has_latin=False, error=None, empty_asr=False)


def make_wd(tmp_path, overrides: dict[int, dict] | None = None, n: int = 10) -> WorkDir:
    """n rows r0..r{n-1}, all passing by default; overrides[i] changes row i."""
    wd = WorkDir(tmp_path)
    rows = []
    for i in range(n):
        r = {**BASE, "id": f"r{i}", **(overrides or {}).get(i, {})}
        r["text"] = r["text"] if "text" in (overrides or {}).get(i, {}) else f"{BASE['text']} {i}"
        rows.append(r)
    df = pd.DataFrame(rows)
    StepFile.write(
        wd.clips_path,
        pd.DataFrame({"id": df["id"], "src_path": "x.wav", "text_raw": df["text"], "src_split": df["split"]}),
        step="import", version=1, params={},
    )
    StepFile.write(
        wd.step_path("text"),
        pd.DataFrame({
            "id": df["id"], "text_clean": df["text"], "has_latin": df["has_latin"], "has_digits": False,
            "n_chars": df["text"].str.len(), "empty_asr": df["empty_asr"], "error": df["error"],
        }),
        step="text", version=1, params={},
    )
    StepFile.write(
        wd.step_path("audio_stats"),
        pd.DataFrame({
            "id": df["id"], "dur": df["dur"], "src_sr": 16000, "clip_frac": df["clip_frac"],
            "silence_share": 0.0, "lufs": -23.0, "error": None,
        }),
        step="audio_stats", version=1, params={},
    )
    return wd


def prof(**kw) -> Profile:
    return Profile(name="t", task="asr", **kw)


def steps(result) -> dict[str, dict]:
    return {s["filter"]: s for s in result.report["steps"]}


def test_each_filter_removes_expected_rows(tmp_path):
    wd = make_wd(tmp_path, {
        0: {"error": "boom"},
        1: {"empty_asr": True},
        2: {"dur": 0.2},
        3: {"dur": 40.0},
        4: {"clip_frac": 0.5},
        5: {"clip_frac": 0.001},  # strict "<": exactly the limit is dropped
        6: {"has_latin": True, "text": "hello"},
    })
    p = prof(
        filters=[
            Range(column="dur", min=0.5, max=30),
            Range(column="clip_frac", max=0.001, max_exclusive=True),
            Flag(column="has_latin", keep=False),
        ],
    )
    res = select(wd, p)
    assert sorted(res.rows["id"]) == ["r7", "r8", "r9"]
    s = res.report["steps"]
    assert [x["filter"] for x in s][:2] == ["not_scored", "step_errors"]
    removed = [x["rows_before"] - x["rows_after"] for x in s]
    assert removed == [0, 2, 2, 2, 1, 0]  # last: train_val_overlap
    assert s[0]["rows_before"] == 10 and s[-1]["rows_after"] == 3
    assert s[2]["filter"].startswith("dur")
    assert s[0]["hours_before"] == pytest.approx((8 * 5.0 + 0.2 + 40.0) / 3600)
    assert (tmp_path / "selection" / "t.parquet").exists()
    assert json.loads((tmp_path / "selection" / "t.report.json").read_text())["steps"] == s


def test_percentile_filter(tmp_path):
    wd = make_wd(tmp_path, {i: {"dur": 1.0 + i} for i in range(10)})
    # cps = n_chars / dur: with a constant text length it is monotone in 1/dur
    res = select(wd, prof(filters=[Percentile(column="cps", lo=10, hi=90)]))
    assert 0 < len(res.rows) < 10
    cps = res.rows["cps"]
    full = select(wd, prof())
    assert cps.min() > full.rows["cps"].min() and cps.max() < full.rows["cps"].max()


def test_drop_latin_profile_flag(tmp_path):
    wd = make_wd(tmp_path, {2: {"has_latin": True, "text": "ok"}})
    res = select(wd, prof(drop_latin=True))
    assert "r2" not in set(res.rows["id"]) and len(res.rows) == 9
    assert "has_latin" in steps(res)


def test_select_empty_result_names_filter(tmp_path):
    wd = make_wd(tmp_path)
    with pytest.raises(SelectError, match="dur"):
        select(wd, prof(filters=[Range(column="dur", min=100)]))


def test_overlap_dropped_from_train_only(tmp_path):
    wd = make_wd(tmp_path, {
        0: {"text": "একই বাক্য।", "split": "val"},
        1: {"text": "একই বাক্য", "split": "train"},   # equal after bn_normalize asr: dropped
        2: {"text": "একই বাক্য", "split": "test"},    # not train: kept
        3: {"text": "একই বাক্য", "split": "val"},     # val duplicate: kept
    })
    res = select(wd, prof())
    ids = set(res.rows["id"])
    assert "r1" not in ids and {"r0", "r2", "r3"} <= ids
    assert res.report["overlap_dropped"] == 1


def test_unexpected_chars_listed(tmp_path):
    wd = make_wd(tmp_path, {4: {"text": "দাম 5€"}})
    res = select(wd, prof())
    unexpected = {c["char"]: c for c in res.report["unexpected_chars"]}
    assert set(unexpected) == {"€"}
    assert "r4" in unexpected["€"]["examples"] and unexpected["€"]["codepoint"] == "U+20AC"
    inv = {c["char"]: c for c in res.report["chars"]}
    assert inv["€"]["count"] == 1 and "আ" in inv and len(inv["আ"]["examples"]) <= 5


def test_profile_missing_step_column(tmp_path):
    wd = make_wd(tmp_path)
    p = load_profile("tts-strict").model_copy(update={"cer_labeller": None})
    p = p.model_copy(update={"filters": [f for f in p.filters if f.column != "cer"]})
    with pytest.raises(SelectError, match="brouhaha|dnsmos"):
        select(wd, p)
    p2 = prof(filters=[Range(column="snr_db", min=25)])
    with pytest.raises(SelectError, match="brouhaha"):
        select(wd, p2)


def test_cer_filter_needs_labeller(tmp_path):
    wd = make_wd(tmp_path)
    with pytest.raises(SelectError, match="--labeller"):
        select(wd, load_profile("asr-v2"))


def test_cer_filter_with_labeller(tmp_path):
    wd = make_wd(tmp_path)
    cer = pd.DataFrame({"id": [f"r{i}" for i in range(10)], "cer": [0.9 if i < 3 else 0.1 for i in range(10)], "error": None})
    StepFile.write(wd.step_path("cer_bangla-conformer"), cer, step="cer_bangla-conformer", version=1, params={})
    res = select(wd, load_profile("asr-v2"), labeller="bangla_conformer")
    assert len(res.rows) == 7
    with pytest.raises(SelectError, match="cer_other"):
        select(wd, load_profile("asr-v2"), labeller="other")


def test_min_speaker_minutes(tmp_path):
    wd = make_wd(tmp_path, {i: {"dur": 60.0} for i in range(10)})
    spk = pd.DataFrame({"id": [f"r{i}" for i in range(10)], "speaker": ["a"] * 6 + ["b"] * 4})
    StepFile.write(wd.step_path("speakers"), spk, step="speakers", version=1, params={})
    res = select(wd, prof(min_speaker_minutes=5))  # a = 6 min kept, b = 4 min dropped
    assert set(res.rows["speaker"]) == {"a"} and len(res.rows) == 6
    assert steps(res)["min_speaker_minutes"]["rows_after"] == 6


def test_voice_flags_are_not_filters(tmp_path):
    p = load_profile("voice")
    assert {f.column for f in p.flags} == {"cer", "dnsmos_ovrl", "snr_db", "max_pause_s"}
    wd = make_wd(tmp_path)
    StepFile.write(  # only the segments are needed (speech_span_s); no cer/dnsmos files
        wd.step_path("brouhaha"),
        pd.DataFrame({"id": [f"r{i}" for i in range(10)], "snr_db": 20.0, "c50_db": 50.0, "max_pause_s": 0.1,
                      "speech_segments": [[[0.5, 4.0]]] * 10, "error": None}),
        step="brouhaha", version=1, params={},
    )
    assert len(select(wd, p).rows) == 10


def test_stale_step_file_refused(tmp_path):
    wd = make_wd(tmp_path)
    old = pd.read_parquet(wd.step_path("audio_stats"))
    StepFile.write(wd.step_path("audio_stats"), old, step="audio_stats", version=0, params={})
    with pytest.raises(StaleStepError, match="audio_stats"):
        select(wd, prof())
    StepFile.write(wd.step_path("audio_stats"), old, step="audio_stats", version=1, params={"x": 1})
    with pytest.raises(StaleStepError, match="audio_stats"):
        select(wd, prof())


def test_shipped_profiles_load():
    names = ["asr-v1", "asr-v2", "tts-loose", "tts-strict", "voice"]
    ps = {n: load_profile(n) for n in names}
    assert all(p.name == n for n, p in ps.items())

    def f(p, col):
        return next(x for x in p.filters if x.column == col)

    assert f(ps["tts-strict"], "dnsmos_ovrl").min == 3.2
    assert f(ps["asr-v2"], "cer").max == 0.5
    assert not any(x.column == "cer" for x in ps["asr-v1"].filters)
    assert f(ps["asr-v1"], "cps").lo == 0.5 and f(ps["asr-v1"], "cps").hi == 99.5
    assert f(ps["tts-loose"], "cps").lo == 1 and f(ps["tts-loose"], "cps").hi == 99
    assert f(ps["tts-loose"], "clip_frac").max == 0.001 and f(ps["tts-loose"], "clip_frac").max_exclusive
    assert (f(ps["voice"], "dur").min, f(ps["voice"], "dur").max) == (1, 20)
    assert all(f(ps[n], "speech_span_s").min == 0.5 for n in ["tts-loose", "tts-strict", "voice"])
    assert (f(ps["tts-strict"], "c50_db").min, f(ps["tts-strict"], "snr_db").min) == (30, 25)
    assert (ps["tts-loose"].min_speaker_minutes, ps["tts-strict"].min_speaker_minutes) == (5, 10)
    assert ps["tts-loose"].drop_latin and ps["tts-strict"].drop_latin and not ps["voice"].drop_latin
    assert all(p.cer_labeller is None for p in ps.values())


def test_select_drops_short_speech_span(tmp_path):
    wd = make_wd(tmp_path, n=2)
    StepFile.write(
        wd.step_path("brouhaha"),
        pd.DataFrame({
            "id": ["r0", "r1"], "snr_db": 30.0, "c50_db": 50.0, "max_pause_s": 0.1,
            # r0 trims to 0.1999 + 2 * 0.15 = 0.4999 s (< 0.5); r1 to 3.0 + 0.3 s
            "speech_segments": [[[1.0, 1.1999]], [[1.0, 4.0]]], "error": None,
        }),
        step="brouhaha", version=1, params={},
    )
    p = Profile(name="t", task="tts", filters=[Range(column="speech_span_s", min=0.5)])
    res = select(wd, p)
    assert list(res.rows["id"]) == ["r1"]
    assert steps(res)["speech_span_s in [0.5, inf]"]["rows_after"] == 1


def test_load_profile_ignores_same_named_dir_in_cwd(tmp_path, monkeypatch):
    (tmp_path / "voice").mkdir()  # e.g. an audio dir named like the shipped profile
    (tmp_path / "asr-v1").mkdir()
    monkeypatch.chdir(tmp_path)
    assert load_profile("voice").name == "voice"
    assert load_profile("asr-v1").name == "asr-v1"


def test_load_profile_by_path(tmp_path):
    y = tmp_path / "mine.yaml"
    y.write_text("name: mine\ntask: asr\nfilters:\n  - {kind: range, column: dur, min: 2}\n")
    assert load_profile(str(y)).filters[0].min == 2


def test_override_relaxes_range_and_percentile_and_is_reported(tmp_path):
    wd = make_wd(tmp_path, {0: {"dur": 0.2}, 1: {"dur": 40.0}})
    p = prof(filters=[Range(column="dur", min=0.5, max=30), Percentile(column="cps", lo=10, hi=90)])
    base = select(wd, p, write=False)
    res = select(wd, p, write=False, overrides=["dur.min=0.1", "dur.max=60", "cps.lo=0", "cps.hi=100"])
    assert len(res.rows) > len(base.rows) and {"r0", "r1"} <= set(res.rows["id"])
    assert res.report["overrides"] == ["dur.min=0.1", "dur.max=60", "cps.lo=0", "cps.hi=100"]
    assert p.filters[0].min == 0.5  # the loaded profile is not mutated


def test_override_unknown_column_or_bound_errors(tmp_path):
    wd = make_wd(tmp_path)
    p = prof(filters=[Range(column="dur", min=0.5, max=30)])
    for bad in ["snr_db.min=1", "dur.lo=1", "dur.min=abc", "dur=3", "dur.min"]:
        with pytest.raises(SelectError):
            select(wd, p, write=False, overrides=[bad])


def test_override_min_speaker_minutes(tmp_path):
    wd = make_wd(tmp_path)
    StepFile.write(
        wd.step_path("speakers"),
        pd.DataFrame({"id": [f"r{i}" for i in range(10)], "speaker": "s0", "error": None}),
        step="speakers", version=1, params={},
    )
    p = prof(min_speaker_minutes=60)
    with pytest.raises(SelectError):
        select(wd, p, write=False)
    assert len(select(wd, p, write=False, overrides=["min_speaker_minutes=0"]).rows) == 10


def test_cli_override_flag(tmp_path):
    from dataprep.__main__ import main

    make_wd(tmp_path, {0: {"dur": 0.2}})
    assert main(["select", "--work", str(tmp_path), "--profile", "asr-v1", "--override", "dur.min=0.1", "--override", "cps.lo=0"]) == 0
    rep = json.loads((tmp_path / "selection" / "asr-v1.report.json").read_text(encoding="utf-8"))
    assert rep["overrides"] == ["dur.min=0.1", "cps.lo=0"]
    assert main(["select", "--work", str(tmp_path), "--profile", "asr-v1", "--override", "nope.min=1"]) == 1


def _cer_file(wd: WorkDir, ids: list[str], errors: dict[str, str] | None = None) -> None:
    errors = errors or {}
    cer = pd.DataFrame({"id": ids, "cer": [None if i in errors else 0.1 for i in ids],
                        "error": [errors.get(i) for i in ids]})
    StepFile.write(wd.step_path("cer_lab"), cer, step="cer_lab", version=1, params={})


CPS_LABEL = "cps within p10-p90"


def test_percentile_bounds_fixed_population(tmp_path):
    """Bounds come from the rows with valid text + audio_stats, whatever later step files cover."""
    n = 20
    wd = make_wd(tmp_path, {i: {"dur": 1.0 + i} for i in range(n)}, n=n)
    pct = Percentile(column="cps", lo=10, hi=90)
    alone = select(wd, prof(filters=[pct]), write=False)
    ids = [f"r{i}" for i in range(n)]
    _cer_file(wd, ids[: n // 2])  # a partial later-step file: the first half only (the high-cps rows)
    partial = select(wd, prof(filters=[Range(column="cer", max=0.5), pct]), labeller="lab", write=False)
    a, b = steps(alone)[CPS_LABEL], steps(partial)[CPS_LABEL]
    assert (a["lo_value"], a["hi_value"]) == (b["lo_value"], b["hi_value"])
    full = pd.Series([len(f"{BASE['text']} {i}") / (1.0 + i) for i in range(n)])
    assert a["lo_value"] == pytest.approx(full.quantile(0.1)) and a["hi_value"] == pytest.approx(full.quantile(0.9))
    # the partial run keeps exactly the scored rows that pass the population-wide bounds
    assert set(partial.rows["id"]) == set(alone.rows["id"]) & set(ids[: n // 2])


def test_percentile_bounds_ignore_errored_rows(tmp_path):
    wd = make_wd(tmp_path, {0: {"dur": 0.01, "error": "boom"}, 1: {"dur": 0.02, "empty_asr": True}})
    res = select(wd, prof(filters=[Percentile(column="cps", lo=0, hi=100)]), write=False)
    s = steps(res)["cps within p0-p100"]
    assert s["hi_value"] == pytest.approx(max(len(f"{BASE['text']} {i}") / 5.0 for i in range(2, 10)))
    assert isinstance(s["lo_value"], float)
    json.dumps(res.report, allow_nan=False)  # plain JSON numbers


def test_not_scored_and_step_errors_counted_separately(tmp_path):
    wd = make_wd(tmp_path, {0: {"error": "boom"}, 1: {"empty_asr": True}})
    ids = [f"r{i}" for i in range(10)]
    _cer_file(wd, ids[:7], errors={"r2": "decode failed"})  # r7..r9 not scored, r2 errored
    res = select(wd, prof(filters=[Range(column="cer", max=0.5)]), labeller="lab", write=False)
    s = steps(res)
    assert s["not_scored"]["rows_before"] - s["not_scored"]["rows_after"] == 3
    assert s["not_scored"]["hours_before"] - s["not_scored"]["hours_after"] == pytest.approx(3 * 5.0 / 3600)
    assert s["step_errors"]["rows_before"] - s["step_errors"]["rows_after"] == 3  # r0, r1 (empty_asr), r2
    assert s["step_errors"]["hours_before"] - s["step_errors"]["hours_after"] == pytest.approx(3 * 5.0 / 3600)
    assert sorted(res.rows["id"]) == ["r3", "r4", "r5", "r6"]
