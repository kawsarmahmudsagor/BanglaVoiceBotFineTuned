from __future__ import annotations

import pyarrow.parquet as pq

from mlops_api.datasets.ingest import ingest


def test_audio_zip_with_top_folder_and_bangla_names(tmp_path, make_zip, audio):  # Review Focus 1
    z = make_zip(top="My Dataset", files={"audio/নমুনা ১.wav": audio.sine(2.0)},
                 metadata="file,text\naudio/নমুনা ১.wav,আমার অর্ডার\n")
    result = ingest(z, "asr", tmp_path / "out")
    assert result.report.ok, result.report.errors
    assert result.n_samples == 1
    table = pq.read_table(result.version_dir / "manifest.parquet").to_pylist()
    assert table[0]["file"] == "audio/নমুনা ১.wav"
    assert table[0]["text"] == "আমার অর্ডার"
    assert table[0]["split"] in {"train", "val", "test"}
    assert (result.version_dir / "files" / "audio" / "নমুনা ১.wav").is_file()
    assert result.report.stats["splits"][table[0]["split"]] == 1


def test_metadata_csv_with_bom_and_crlf(tmp_path, make_zip, audio):  # Review Focus 2
    z = make_zip(files={"audio/a.wav": audio.sine(2.0)},
                 metadata="\ufefffile,text\r\naudio/a.wav,হ্যালো\r\n")
    result = ingest(z, "asr", tmp_path / "out")
    assert result.report.ok, result.report.errors
    rows = pq.read_table(result.version_dir / "manifest.parquet").to_pylist()
    assert rows[0]["file"] == "audio/a.wav" and rows[0]["text"] == "হ্যালো"


def test_ingest_same_content_same_hash_and_errors_write_nothing(tmp_path, make_zip, audio):
    out = tmp_path / "out"
    meta = "file,text\na.wav,ক\n"
    r1 = ingest(make_zip(files={"a.wav": audio.sine(2.0)}, metadata=meta, name="1.zip"), "asr", out)
    r2 = ingest(make_zip(files={"a.wav": audio.sine(2.0)}, metadata=meta, top="wrapped", name="2.zip"), "asr", out)
    assert r1.version_hash == r2.version_hash and len(r1.version_hash) == 16
    r3 = ingest(make_zip(files={"a.wav": audio.sine(2.0, freq=300)}, metadata=meta, name="3.zip"), "asr", out)
    assert r3.version_hash != r1.version_hash
    before = sorted(p.name for p in out.iterdir())

    bad = ingest(make_zip(files={}, metadata="file,text\nmissing.wav,ক\n", name="bad.zip"), "asr", out)
    assert not bad.report.ok and bad.version_hash is None and bad.version_dir is None and bad.n_samples == 0
    assert sorted(p.name for p in out.iterdir()) == before  # no new version, staging cleaned up


def test_llm_ingest_writes_manifest_without_files(tmp_path, jsonl):
    p = jsonl([{"messages": [{"role": "user", "content": "প্রশ্ন"},
                             {"role": "assistant", "content": "উত্তর"}]}])
    result = ingest(p, "llm-sft", tmp_path / "out")
    assert result.report.ok and result.n_samples == 1
    rows = pq.read_table(result.version_dir / "manifest.parquet").to_pylist()
    assert {"sample_id", "messages_json", "tools_json", "split"} <= set(rows[0])
    assert not (result.version_dir / "files").exists()


def test_format_mismatch_is_error_issue_not_exception(tmp_path, make_zip, jsonl, audio):
    z = make_zip(files={"a.wav": audio.sine(1.0)}, metadata="file,text\na.wav,ক\n")
    r = ingest(z, "llm-sft", tmp_path / "out")
    assert [i.code for i in r.report.errors] == ["unsupported_format"] and r.version_hash is None
    r = ingest(jsonl([{"messages": []}]), "asr", tmp_path / "out")
    assert [i.code for i in r.report.errors] == ["unsupported_format"]


def test_zip_path_traversal_is_rejected(tmp_path, make_zip, audio):
    out = tmp_path / "out"
    for evil in ("../evil.wav", "/abs/evil.wav", "C:/evil.wav", "a\\..\\..\\evil.wav"):
        z = make_zip(files={"a.wav": audio.sine(1.0)}, metadata="file,text\na.wav,ক\n",
                     extra_members={evil: b"x"}, name="evil.zip")
        r = ingest(z, "asr", out)
        assert [i.code for i in r.report.errors] == ["unsafe_path"], evil
        assert r.version_dir is None
    assert not (tmp_path / "evil.wav").exists()
    assert [p for p in out.iterdir()] == []


def test_not_a_zip_is_error_issue(tmp_path):
    f = tmp_path / "x.zip"
    f.write_bytes(b"nope")
    r = ingest(f, "asr", tmp_path / "out")
    assert [i.code for i in r.report.errors] == ["unsupported_format"]
