from __future__ import annotations

import json
import zipfile

import pyarrow.parquet as pq

from app.datasets.ingest import ingest


def test_audio_zip_with_top_folder_and_bangla_names(tmp_path, make_zip, audio):  # Review Focus 1
    z = make_zip(top="My Dataset", files={"audio/নমুনা ১.wav": audio.sine(2.0)},
                 metadata="file,text\naudio/নমুনা ১.wav,আমার অর্ডার\n")
    result = ingest(z, "asr", tmp_path / "out")
    assert result.report.ok, result.report.errors
    assert result.n_samples == 1
    table = pq.read_table(result.version_dir / "manifest.parquet").to_pylist()
    assert table[0]["file"] == "audio/নমুনা ১.wav"
    assert table[0]["text"] == "আমার অর্ডার"
    assert (result.version_dir / "files" / "audio" / "নমুনা ১.wav").is_file()
    assert table[0]["split"] is None
    assert result.report.stats["given_splits"] == {"train": 0, "val": 0, "test": 0, "unassigned": 1}


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
    r = ingest(z, "llm-sft", tmp_path / "out")  # a zip is now valid llm-sft input, but has no split files
    assert [i.code for i in r.report.errors] == ["no_split_files"] and r.version_hash is None
    bad = tmp_path / "x.txt"
    bad.write_text("x")
    r = ingest(bad, "llm-sft", tmp_path / "out")
    assert [i.code for i in r.report.errors] == ["unsupported_format"]
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


def test_duplicate_file_rows_are_an_error_not_a_crash(tmp_path, make_zip, audio):
    z = make_zip(files={"a.wav": audio.sine(2.0)}, metadata="file,text\na.wav,ক\na.wav,খ\n")
    r = ingest(z, "asr", tmp_path / "out")
    assert [i.code for i in r.report.errors] == ["duplicate_file"]
    assert "a.wav" in r.report.errors[0].message
    assert r.version_dir is None and r.n_samples == 0


def test_optional_columns_absent_from_first_row_are_kept(tmp_path, make_zip, audio):
    z = make_zip(files={"a.wav": audio.sine(2.0), "b.wav": audio.sine(2.0, freq=300)},
                 metadata="file,text,note\na.wav,ক\nb.wav,খ,hello\n")
    r = ingest(z, "asr", tmp_path / "out")
    assert r.report.ok, r.report.errors
    rows = pq.read_table(r.version_dir / "manifest.parquet").to_pylist()
    assert {x["text"]: x["note"] for x in rows} == {"ক": None, "খ": "hello"}


def test_zero_sample_uploads_are_errors(tmp_path, make_zip, jsonl):
    r = ingest(jsonl([]), "llm-sft", tmp_path / "out")
    assert [i.code for i in r.report.errors] == ["no_samples"] and r.version_dir is None
    r = ingest(make_zip(files={}, metadata="file,text\n"), "asr", tmp_path / "out")
    assert [i.code for i in r.report.errors] == ["no_samples"] and r.version_dir is None


def test_upload_suffix_is_case_insensitive(tmp_path, make_zip, jsonl, audio):
    p = jsonl([{"messages": [{"role": "user", "content": "প্রশ্ন"},
                             {"role": "assistant", "content": "উত্তর"}]}])
    up = p.rename(p.with_name("DATA.JSONL"))
    assert ingest(up, "llm-sft", tmp_path / "out").report.ok
    z = make_zip(files={"a.wav": audio.sine(2.0)}, metadata="file,text\na.wav,ক\n")
    z = z.rename(z.with_name("UPLOAD.ZIP"))
    assert ingest(z, "asr", tmp_path / "out2").report.ok


def test_zip_larger_than_free_space_minus_reserve_is_insufficient_disk(tmp_path, make_zip, audio, monkeypatch):
    import collections

    from app.datasets import ingest as ingest_mod

    z = make_zip(files={"a.wav": audio.sine(1.0)}, metadata="file,text\na.wav,ক\n")
    usage = collections.namedtuple("usage", "total used free")
    gb = 1024**3
    # 10 GB reserve, 10 GB + a few bytes free: the uncompressed size does not fit
    monkeypatch.setattr(ingest_mod.shutil, "disk_usage", lambda p: usage(100 * gb, 0, 10 * gb + 100))
    r = ingest(z, "asr", tmp_path / "out", disk_min_gb=10)
    assert [i.code for i in r.report.errors] == ["insufficient_disk"] and r.version_dir is None
    assert not list((tmp_path / "out").glob("*"))
    # plenty of space: same archive ingests
    monkeypatch.setattr(ingest_mod.shutil, "disk_usage", lambda p: usage(100 * gb, 0, 50 * gb))
    assert ingest(z, "asr", tmp_path / "out", disk_min_gb=10).report.ok


def _manifest(result):
    return pq.read_table(result.version_dir / "manifest.parquet").to_pylist()


def _conv(t):
    return {"messages": [{"role": "user", "content": t}, {"role": "assistant", "content": t + "!"}]}


def _jl(*items):
    return "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items)


def test_llm_split_field(tmp_path, jsonl):
    r = ingest(jsonl([{**_conv("ক"), "split": "dev"}, _conv("খ")]), "llm-sft", tmp_path / "out")
    assert r.report.ok
    assert sorted(x["split"] or "" for x in _manifest(r)) == ["", "val"]


def test_llm_zip_file_layout(tmp_path):
    z = tmp_path / "u.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("train.jsonl", _jl(_conv("ক")))
        zf.writestr("validation.jsonl", _jl(_conv("খ")))
    r = ingest(z, "llm-sft", tmp_path / "out")
    assert r.report.ok, r.report.errors
    assert sorted(x["split"] for x in _manifest(r)) == ["train", "val"]


def test_llm_zip_without_split_files_errors(tmp_path):
    z = tmp_path / "u.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("data.jsonl", _jl(_conv("ক")))
    r = ingest(z, "llm-sft", tmp_path / "out")
    assert [i.code for i in r.report.errors] == ["no_split_files"] and r.version_dir is None


def test_audio_split_column(tmp_path, make_zip, audio):
    z = make_zip(files={"a.wav": audio.sine(2.0), "b.wav": audio.sine(2.0, freq=300), "c.wav": audio.sine(2.0, freq=400)},
                 metadata="file,text,split\na.wav,ক,Training\nb.wav,খ,test\nc.wav,গ,\n".replace("split\n", "split" + "\n"))
    r = ingest(z, "asr", tmp_path / "out")
    assert r.report.ok, r.report.errors
    assert sorted(x["split"] or "" for x in _manifest(r)) == ["", "test", "train"]


def test_audio_metadata_split_files(tmp_path, make_zip, audio):
    z = make_zip(files={"audio/a.wav": audio.sine(2.0), "audio/b.wav": audio.sine(2.0, freq=300)}, metadata=None,
                 extra_members={"metadata_train.csv": "file,text\naudio/a.wav,ক\n".encode(),
                                "metadata_valid.csv": "file,text\naudio/b.wav,খ\n".encode()})
    r = ingest(z, "asr", tmp_path / "out")
    assert r.report.ok, r.report.errors
    assert {x["file"]: x["split"] for x in _manifest(r)} == {"audio/a.wav": "train", "audio/b.wav": "val"}


def test_audio_mixed_metadata_layouts_errors(tmp_path, make_zip, audio):
    z = make_zip(files={"a.wav": audio.sine(2.0)}, metadata="file,text\na.wav,ক\n",
                 extra_members={"metadata_test.csv": "file,text\na.wav,ক\n".encode()})
    r = ingest(z, "asr", tmp_path / "out")
    assert [i.code for i in r.report.errors] == ["mixed_metadata"]


def test_audio_split_value_in_split_file_errors(tmp_path, make_zip, audio):
    z = make_zip(files={"a.wav": audio.sine(2.0)}, metadata=None,
                 extra_members={"metadata_train.csv": "file,text,split\na.wav,ক,test\n".encode()})
    r = ingest(z, "asr", tmp_path / "out")
    assert [i.code for i in r.report.errors] == ["split_in_split_file"]


def test_unknown_split_value_errors(tmp_path, make_zip, audio):
    z = make_zip(files={"a.wav": audio.sine(2.0), "b.wav": audio.sine(2.0, freq=300)},
                 metadata="file,text,split\na.wav,ক,train\nb.wav,খ,hold\n")
    r = ingest(z, "asr", tmp_path / "out")
    assert [i.message for i in r.report.errors] == ["split value 'hold' on line 3: use train, valid or test"]
    assert r.version_dir is None


def test_split_source_and_given_splits_recorded(tmp_path, jsonl):
    r = ingest(jsonl([{**_conv("ক"), "split": "train"}, {**_conv("খ"), "split": "test"}, _conv("গ")]),
               "llm-sft", tmp_path / "out")
    assert r.report.stats["split_source"] == "dataset"
    assert r.report.stats["given_splits"] == {"train": 1, "val": 0, "test": 1, "unassigned": 1}
    r2 = ingest(jsonl([_conv("ঘ")], name="n.jsonl"), "llm-sft", tmp_path / "out")
    assert r2.report.stats["split_source"] == "none"
