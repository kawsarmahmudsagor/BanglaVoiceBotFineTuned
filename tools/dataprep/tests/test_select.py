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
    assert [x["filter"] for x in s][0] == "step_errors"
    removed = [x["rows_before"] - x["rows_after"] for x in s]
    assert removed == [2, 2, 2, 1, 0]  # last: train_val_overlap
    assert s[0]["rows_before"] == 10 and s[-1]["rows_after"] == 3
    assert s[1]["filter"].startswith("dur")
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


def test_load_profile_by_path(tmp_path):
    y = tmp_path / "mine.yaml"
    y.write_text("name: mine\ntask: asr\nfilters:\n  - {kind: range, column: dur, min: 2}\n")
    assert load_profile(str(y)).filters[0].min == 2
