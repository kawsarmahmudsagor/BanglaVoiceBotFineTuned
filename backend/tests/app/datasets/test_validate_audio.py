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
