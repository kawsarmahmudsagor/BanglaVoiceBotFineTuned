from __future__ import annotations

from app.datasets.validate_audio import validate_audio


def codes(issues) -> set[str]:
    return {i.code for i in issues}


def test_audio_checks(dir_with, audio):
    root = dir_with(
        {"short.wav": audio.sine(0.2), "loud.wav": audio.clipped(2.0),
         "quiet.wav": audio.silence(2.0), "sr.wav": audio.sine(2.0, sr=8000)},
        "file,text\nshort.wav,এক\nloud.wav,দুই\nquiet.wav,তিন\nsr.wav,চার\n",
    )
    report, rows = validate_audio(root, "asr")
    assert report.ok and len(rows) == 4
    assert codes(report.warnings) >= {"duration_out_of_range", "clipping", "high_silence",
                                      "mixed_sample_rates", "loudness_out_of_range"}
    by_code = {i.code: i.sample_id for i in report.warnings if i.sample_id}
    assert by_code["clipping"] in {r["sample_id"] for r in rows}
    assert report.stats["n_samples"] == 4
    assert abs(report.stats["total_duration_s"] - 6.2) < 0.01


def test_clean_audio_has_no_warnings_and_expected_row(dir_with, audio):
    root = dir_with({"a.wav": audio.sine(2.0, amp=0.1)}, "file,text,note\na.wav,হ্যালো,x\n")
    report, rows = validate_audio(root, "asr")
    assert report.ok and not report.warnings, report.warnings
    r = rows[0]
    assert (r["file"], r["text"], r["sample_rate"], r["note"]) == ("a.wav", "হ্যালো", 16000, "x")
    assert abs(r["duration_s"] - 2.0) < 1e-3


def test_audio_errors(dir_with, audio, tmp_path):
    # missing metadata.csv
    report, _ = validate_audio(dir_with({"a.wav": audio.sine(1.0)}, None, "d1"), "asr")
    assert codes(report.errors) == {"missing_metadata"}
    # tts without speaker column
    report, _ = validate_audio(dir_with({"a.wav": audio.sine(1.0)}, "file,text\na.wav,ক\n", "d2"), "tts")
    assert codes(report.errors) == {"missing_column"} and "speaker" in report.errors[0].message
    # row pointing at an absent file
    report, _ = validate_audio(dir_with({}, "file,text\nnope.wav,ক\n", "d3"), "asr")
    assert codes(report.errors) == {"missing_audio"}
    # wav without a row
    report, _ = validate_audio(
        dir_with({"a.wav": audio.sine(1.0), "b.wav": audio.sine(1.0)}, "file,text\na.wav,ক\n", "d4"), "asr")
    assert codes(report.errors) == {"missing_transcript"} and "b.wav" in report.errors[0].message
    # empty transcript
    report, _ = validate_audio(dir_with({"a.wav": audio.sine(1.0)}, "file,text\na.wav,\n", "d5"), "asr")
    assert codes(report.errors) == {"missing_transcript"}
    # corrupt wav
    report, _ = validate_audio(dir_with({"a.wav": b"RIFFnotreallyawav"}, "file,text\na.wav,ক\n", "d6"), "asr")
    assert codes(report.errors) == {"undecodable"}


def test_tts_requires_speaker_and_keeps_it(dir_with, audio):
    root = dir_with({"a.wav": audio.sine(1.0, amp=0.1)}, "file,text,speaker\na.wav,ক,spk1\n")
    report, rows = validate_audio(root, "tts")
    assert report.ok and rows[0]["speaker"] == "spk1"


def _clipped_dataset(dir_with, audio, n):
    files = {f"c{i:04d}.wav": audio.clipped(0.1) for i in range(n)}
    rows = "".join(f"c{i:04d}.wav,ক{i}\n" for i in range(n))
    return dir_with(files, "file,text\n" + rows)


def test_warning_cap(dir_with, audio):
    root = _clipped_dataset(dir_with, audio, 1200)
    report, rows = validate_audio(root, "asr", workers=1)
    assert len(rows) == 1200
    assert sum(w.code == "clipping" for w in report.warnings) == 1000
    assert report.stats["warning_counts"]["clipping"] == 1200


def test_report_add_warning_cap():
    from app.datasets.report import Issue, Report
    r = Report()
    for i in range(5):
        r.add_warning(Issue("x", str(i)), cap=3)
    assert [w.message for w in r.warnings] == ["0", "1", "2"]
    assert r.stats["warning_counts"] == {"x": 5}


def test_parallel_matches_serial(dir_with, audio):
    files = {f"a{i}.wav": (audio.clipped(0.6) if i % 3 == 0 else audio.sine(0.7 + i * 0.01, amp=0.1))
             for i in range(12)}
    files["bad.wav"] = b"RIFFnotreallyawav"
    files["hz8.wav"] = audio.sine(1.0, sr=8000)
    meta = "file,text\n" + "".join(f"a{i}.wav,ক{i}\n" for i in range(12)) + "bad.wav,খ\nhz8.wav,গ\nnope.wav,ঘ\n"
    root = dir_with(files, meta)
    serial = validate_audio(root, "asr", workers=1)
    parallel = validate_audio(root, "asr", workers=4)
    assert serial[1] == parallel[1] and len(serial[1]) == 13
    assert serial[0] == parallel[0]
    assert codes(serial[0].errors) == {"undecodable", "missing_audio"}
