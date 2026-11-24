from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from dataprep import check_voice
from dataprep.__main__ import main
from dataprep.select import cer_step_name
from dataprep.sources.voice import import_voice
from dataprep.steps import audio_stats, text
from dataprep.workdir import StepFile, WorkDir
from fixtures import make_voice

SPEC = "whisper-hf:fake"
TEXTS = ["আমি ভাত খাই", "ক্ষমা করবেন", "আপনার অর্ডার নম্বর", "জি", "ঠিক আছে", "ধন্যবাদ", "শুভ সকাল", "আবার বলুন"]


def make_wd(tmp_path, *, session=False, **per_take) -> WorkDir:
    """8 takes t0..t7 imported for real; scores written by hand. per_take: {"t1": {"cer": 0.5}, ...}."""
    rows = [{"name": f"t{i}", "text": TEXTS[i]} for i in range(8)]
    rows[5]["amp"] = 3.0  # clipped
    rows[6]["sr"] = 44100  # wrong sample rate
    if session:
        for i, r in enumerate(rows):
            r["session"] = "s1" if i < 4 else "s2"
    src = make_voice(tmp_path / "src", rows)
    wd = WorkDir(tmp_path / "wd")
    import_voice(src, wd)
    text.run(wd)
    audio_stats.run(wd)
    ids = [f"t{i}" for i in range(8)]
    over = lambda col, default: [per_take.get(i, {}).get(col, default) for i in ids]  # noqa: E731
    StepFile.write(
        wd.step_path("dnsmos"),
        pd.DataFrame({"id": ids, "dnsmos_p808": 4.0, "dnsmos_sig": 4.0, "dnsmos_bak": 4.0,
                      "dnsmos_ovrl": over("dnsmos_ovrl", 3.9), "error": None}),
        step="dnsmos", version=1, params={},
    )
    StepFile.write(
        wd.step_path("brouhaha"),
        pd.DataFrame({"id": ids, "snr_db": over("snr_db", 40.0), "c50_db": 50.0, "max_pause_s": over("max_pause_s", 0.3),
                      "speech_segments": [[[0.1, 0.9]]] * 8, "error": None}),
        step="brouhaha", version=1, params={},
    )
    StepFile.write(
        wd.step_path(cer_step_name(SPEC)),
        pd.DataFrame({"id": ids, "hyp": "x", "cer": over("cer", 0.0), "error": None}),
        step=cer_step_name(SPEC), version=1, params={},
    )
    return wd


def flags_of(report: dict) -> dict[str, set[str]]:
    return {t["id"]: {f["column"] for f in t["flags"]} for t in report["flagged"]}


def test_flags_raised_for_planted_bad_takes(tmp_path):
    wd = make_wd(tmp_path, t1={"cer": 0.5}, t2={"dnsmos_ovrl": 3.0}, t3={"snr_db": 20.0}, t4={"max_pause_s": 2.0})
    report = check_voice.build(wd, SPEC, coverage_path=None)
    flags = flags_of(report)
    assert flags["t1"] == {"cer"} and flags["t2"] == {"dnsmos_ovrl"} and flags["t3"] == {"snr_db"}
    assert flags["t4"] == {"max_pause_s"}
    assert flags["t5"] == {"clip_frac"}
    assert set(flags) == {"t1", "t2", "t3", "t4", "t5", "t6"}  # t0 and t7 are clean


def test_thresholds_come_from_the_voice_profile(tmp_path):
    # exactly at a threshold passes: cer <= 0.02, dnsmos_ovrl >= 3.5, snr_db >= 35, max_pause_s <= 1.0
    wd = make_wd(tmp_path, t0={"cer": 0.02, "dnsmos_ovrl": 3.5, "snr_db": 35.0, "max_pause_s": 1.0})
    assert "t0" not in flags_of(check_voice.build(wd, SPEC, coverage_path=None))


def test_inconsistent_format_is_flagged(tmp_path):
    wd = make_wd(tmp_path)
    report = check_voice.build(wd, SPEC, coverage_path=None)
    fmt = report["format"]
    assert fmt["expected"] == {"sample_rate": 48000, "bit_depth": 24}
    assert fmt["consistent"] is False
    assert [t["id"] for t in fmt["inconsistent"]] == ["t6"]
    assert fmt["inconsistent"][0]["sample_rate"] == 44100
    assert "format" in flags_of(report)["t6"]
    assert fmt["counts"] == {"48000 Hz / 24-bit": 7, "44100 Hz / 24-bit": 1}


def test_hours_and_speech_rate_per_session(tmp_path):
    wd = make_wd(tmp_path, session=True)
    report = check_voice.build(wd, SPEC, coverage_path=None)
    assert report["n_takes"] == 8
    assert report["hours"] == pytest.approx(8 / 3600, rel=1e-3)
    rate = report["speech_rate"]["sessions"]
    assert set(rate) == {"s1", "s2"} and rate["s1"]["n"] == 4
    assert rate["s1"]["mean_cps"] > 0
    assert "drift" in report["speech_rate"]


def test_session_falls_back_to_mtime_date(tmp_path):
    wd = make_wd(tmp_path)
    clips = pd.read_parquet(wd.clips_path)
    stamp = pd.Timestamp("2026-03-05 12:00").timestamp()
    for p in clips.src_path[:3]:
        os.utime(p, (stamp, stamp))
    report = check_voice.build(wd, SPEC, coverage_path=None)
    sessions = report["speech_rate"]["sessions"]
    assert sessions["2026-03-05"]["n"] == 3 and sum(s["n"] for s in sessions.values()) == 8


def test_coverage_counts_and_reports_targets_below_20(tmp_path):
    wd = make_wd(tmp_path)
    cov = tmp_path / "cov.txt"
    cov.write_text("# comment\n\nক্ষ\nা\nজ্ঞ\n", encoding="utf-8")
    report = check_voice.build(wd, SPEC, coverage_path=cov)
    c = report["coverage"]
    assert c["min_count"] == 20
    assert c["counts"]["ক্ষ"] == 1 and c["counts"]["জ্ঞ"] == 0 and c["counts"]["া"] >= 5
    assert "ক্ষ" in c["below"] and "জ্ঞ" in c["below"]


def test_coverage_counts_nfc(tmp_path):
    assert check_voice.count_target("ক্ষ ক্ষ", "ক্ষ") == 2
    assert check_voice.count_target("ড়", "ড়") == 1  # decomposed text matches the precomposed target


def test_default_coverage_list_is_reasonable():
    targets = check_voice.load_targets(check_voice.DEFAULT_COVERAGE)
    assert 40 <= len(targets) <= 90 and "ক্ষ" in targets and "ি" in targets and len(set(targets)) == len(targets)
    for letter in "খকঘগছচঝজথতধদফপভবঠটঢড":
        assert letter in targets, letter


def test_html_has_player_text_and_flags_per_flagged_take(tmp_path):
    wd = make_wd(tmp_path, t1={"cer": 0.5})
    report = check_voice.build(wd, SPEC, coverage_path=None)
    html = check_voice.render_html(report)
    assert html.count("<audio") == len(report["flagged"]) == 3  # t1, t5, t6
    assert "ক্ষমা করবেন" in html and "cer" in html
    assert 'src="../src/wavs/t1.wav"' in html  # relative to the work dir, which sits beside src/
    assert "prefers-color-scheme: dark" in html


def test_html_escapes_text(tmp_path):
    report = {"flagged": [{"id": "x", "href": "a.wav", "text": "<b>hi</b>", "flags": []}]}
    html = check_voice.render_flagged_rows(report["flagged"])
    assert "<b>hi</b>" not in html and "&lt;b&gt;" in html


def test_write_outputs_json_and_html(tmp_path):
    wd = make_wd(tmp_path, t1={"cer": 0.5})
    report = check_voice.build(wd, SPEC, coverage_path=None)
    check_voice.write_outputs(wd, report)
    data = json.loads((wd.root / "check_voice.json").read_text(encoding="utf-8"))
    assert data["flagged"] == report["flagged"] and data["hours"] == report["hours"]
    assert (wd.root / "check_voice.html").read_text(encoding="utf-8").startswith("<!doctype html>")


def test_run_executes_all_steps_with_injected_models(tmp_path, monkeypatch):
    import numpy as np

    from dataprep.steps import brouhaha, dnsmos, transcribe

    src = make_voice(tmp_path / "src", [{"name": "a", "text": "আমি ভাত খাই"}, {"name": "b", "text": "জি"}])
    wd = WorkDir(tmp_path / "wd")
    import_voice(src, wd)
    monkeypatch.setattr(dnsmos, "_scorer", lambda y: (4.0, 4.0, 4.0, 4.0))
    frames = np.zeros((100, 3))
    frames[10:90, 0], frames[:, 1] = 1.0, 40.0
    monkeypatch.setattr(brouhaha, "_scorer", lambda p: (0.01, frames))
    monkeypatch.setattr(transcribe, "_labeller", lambda paths: ["আমি ভাত খাই" if "a.wav" in str(p) else "জি" for p in paths])
    assert main(["check-voice", "--work", str(wd.root), "--labeller", SPEC]) == 0
    data = json.loads((wd.root / "check_voice.json").read_text(encoding="utf-8"))
    assert data["n_takes"] == 2 and data["flagged"] == []
    assert (wd.root / "check_voice.html").is_file()
