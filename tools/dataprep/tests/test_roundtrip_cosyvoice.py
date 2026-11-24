import json

import pytest

from dataprep import roundtrip_cosyvoice as rt
from dataprep.__main__ import main

ASP_RET = "ঠাকুর খুব ভালো"  # ঠ is aspirated and retroflex; খ, ভ aspirated
ASP_ONLY = "খবর ভালো"
RET_ONLY = "টাকা দাও"
NUKTA = "পড়া ঘর"  # ড় (ড + nukta after NFC) retroflex, ঘ aspirated


def rows(*texts):
    return [{"file": f"wavs/{i}.wav", "text": t} for i, t in enumerate(texts)]


def test_has_contrasts():
    assert rt.has_contrasts(ASP_RET)
    assert rt.has_contrasts(NUKTA)
    assert not rt.has_contrasts(ASP_ONLY)
    assert not rt.has_contrasts(RET_ONLY)


def test_select_filters_text_duration_and_exclude():
    data = rows(ASP_RET, ASP_ONLY, ASP_RET, ASP_RET, NUKTA)
    durations = {"wavs/0.wav": 5.0, "wavs/2.wav": 1.0, "wavs/3.wav": 6.0, "wavs/4.wav": 4.0}
    picked = rt.select_clips(data, 10, durations.get, exclude=["wavs/3.wav"])
    assert {c.file for c in picked} == {"wavs/0.wav", "wavs/4.wav"}  # 1: text, 2: too short, 3: excluded
    assert [c.id for c in picked] == ["00", "01"]


def test_select_is_order_independent_and_stops_at_n():
    data = rows(*[ASP_RET] * 20)
    calls = []

    def dur(f):
        calls.append(f)
        return 5.0

    a = rt.select_clips(data, 5, dur)
    b = rt.select_clips(list(reversed(data)), 5, lambda f: 5.0)
    assert [c.file for c in a] == [c.file for c in b]
    assert len(calls) == 5  # durations read lazily, only until n are found
    assert [c.file for c in rt.select_clips(data, 5, dur, salt="x")] != [c.file for c in a]


def test_select_rejects_max_over_tokenizer_limit():
    with pytest.raises(ValueError):
        rt.select_clips([], 1, lambda f: 1.0, max_s=31)


def test_pick_prompt():
    data = rows("a", "b", "c")
    assert rt.pick_prompt(data, {"wavs/1.wav": 8.0}.get) == "wavs/1.wav"
    assert rt.pick_prompt(data, lambda f: 2.0) is None


def test_clip_cer_normalizes_punctuation_and_spaces():
    assert rt.clip_cer("ঠাকুর, খুব ভালো।", "ঠাকুর খুব  ভালো") == 0.0
    assert rt.clip_cer("খুব", "কুব") == pytest.approx(1 / 3)


def test_cer_gate():
    ok = rt.cer_gate([{"cer_orig": 0.10, "cer_resynth": 0.14}, {"cer_orig": 0.10, "cer_resynth": 0.16}])
    assert ok["delta"] == pytest.approx(0.05) and ok["passed"]
    bad = rt.cer_gate([{"cer_orig": 0.1, "cer_resynth": 0.3}])
    assert not bad["passed"]
    with pytest.raises(ValueError):
        rt.cer_gate([])


def test_listening_gate():
    ids = [f"{i:02d}" for i in range(30)]
    v = {i: True for i in ids}
    for i in ids[:3]:
        v[i] = False
    g = rt.listening_gate(v, ids)
    assert (g["preserved"], g["needed"], g["passed"]) == (27, 27, True)
    v[ids[3]] = False
    assert not rt.listening_gate(v, ids)["passed"]
    with pytest.raises(ValueError, match="05"):
        rt.listening_gate({i: True for i in ids if i != "05"}, ids)


def _summary(cer_passed=True):
    clips = [
        {"id": f"{i:02d}", "file": f"wavs/{i}.wav", "text": "<b>ঠাকুর</b>", "duration": 5.0,
         "hyp_orig": "x", "hyp_resynth": "y", "cer_orig": 0.1, "cer_resynth": 0.12}
        for i in range(30)
    ]
    return {
        "run_id": "r1", "model_dir": "m", "cosyvoice_commit": "abc", "stt_model": "s", "prompt_file": "p.wav",
        "cer_gate": {"mean_cer_orig": 0.1, "mean_cer_resynth": 0.12, "delta": 0.02, "max_delta": 0.05,
                     "passed": cer_passed},
        "clips": clips,
    }


def test_report_escapes_and_lists_every_clip():
    page = rt.render_report(_summary())
    assert "&lt;b&gt;ঠাকুর&lt;/b&gt;" in page and "<b>ঠাকুর</b>" not in page
    assert page.count('data-clip="') == 30
    assert 'data-needed="27"' in page
    assert 'src="resynth/29.wav"' in page


def test_finalize_writes_gate_and_exit_code(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps(_summary()), encoding="utf-8")
    verdicts = {f"{i:02d}": i >= 3 for i in range(30)}  # 27 preserved
    (tmp_path / "verdicts.json").write_text(json.dumps({"preserved": verdicts}), encoding="utf-8")
    assert main(["roundtrip-cosyvoice", "finalize", "--out", str(tmp_path)]) == 0
    gate = json.loads((tmp_path / "gate.json").read_text(encoding="utf-8"))
    assert gate["passed"] and gate["failed_clips"] == ["00", "01", "02"]

    (tmp_path / "summary.json").write_text(json.dumps(_summary(cer_passed=False)), encoding="utf-8")
    assert main(["roundtrip-cosyvoice", "finalize", "--out", str(tmp_path)]) == 1


def test_finalize_missing_verdict_is_an_error(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps(_summary()), encoding="utf-8")
    (tmp_path / "v.json").write_text(json.dumps({"preserved": {"00": True}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="no listening verdict"):
        main(["roundtrip-cosyvoice", "finalize", "--out", str(tmp_path), "--verdicts", str(tmp_path / "v.json")])
