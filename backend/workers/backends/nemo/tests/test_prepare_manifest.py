import json

import numpy as np

from mlops_backend.prepare.manifest import Decoded, build_manifest, manifest_spec


def arow(text, path="/x/a.wav", sid="s", dur=1.0):
    return {"sample_id": sid, "audio_path": path, "text": text, "duration_s": dur}


def ok(path):
    return 1.0


def lines(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]


def test_nemo_skips_unk_rows(tmp_path):
    enc = lambda t: [0 if c != "ক" else 9 for c in t]  # noqa: E731
    res = build_manifest(enc, 9, probe=ok)({"train": [arow("আম"), arow("কম")]}, tmp_path)
    assert res.counts["train"] == 1 and res.skipped[0].reason == "tokenizer maps characters to <unk>: ক"


def test_manifest_references_source_audio(tiny_audio_ds, tmp_path):
    root, names = tiny_audio_ds(sr=44100, n=2)
    out = tmp_path / "out"
    rows = [{"sample_id": f"s{i}", "audio_path": str(root / "files" / n), "text": "আমি  যাব"}
            for i, n in enumerate(names)]
    res = build_manifest(None, None)({"train": rows}, out)
    assert res.counts["train"] == 2
    recs = lines(out / "train.jsonl")
    assert recs[0]["audio_filepath"] == rows[0]["audio_path"]
    assert recs[0]["duration"] == 0.5 and recs[0]["text"] == "আমি যাব"
    assert not (out / "audio").exists()


def test_unreadable_format_transcoded_to_flac(tmp_path):
    import soundfile as sf

    audio = np.zeros(8000, dtype="float32")
    probe = lambda p: Decoded(audio)  # noqa: E731
    res = build_manifest(None, None, probe=probe)({"train": [arow("আ", "/x/a.m4a", "sid1", dur=0)]}, tmp_path)
    flac = tmp_path / "audio" / "sid1.flac"
    assert res.counts["train"] == 1 and flac.is_file()
    info = sf.info(str(flac))
    assert info.samplerate == 16000 and info.channels == 1
    rec = lines(tmp_path / "train.jsonl")[0]
    assert rec["audio_filepath"] == str(flac) and rec["duration"] == 0.5


def test_undecodable_audio_skipped(tmp_path):
    from mlops_backend.prepare import manifest as m

    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    probe = m.make_probe(lambda p, sr: (_ for _ in ()).throw(ValueError("x")))
    good = tmp_path / "g.wav"
    import soundfile as sf

    sf.write(str(good), np.zeros(1600, dtype="float32"), 16000)
    res = build_manifest(None, None, probe=probe)(
        {"train": [arow("আ", str(bad), "b", dur=0), arow("আ", str(good), "g", dur=0)]}, tmp_path / "o")
    assert res.counts["train"] == 1
    assert res.skipped[0].reason == "undecodable audio: ValueError"


def test_long_and_empty_skipped(tmp_path):
    res = build_manifest(None, None, probe=ok)(
        {"train": [arow("আ"), arow("আ", sid="l", dur=31.0), arow("  ", sid="e")]}, tmp_path)
    assert [s.reason for s in res.skipped] == ["duration 31.0 s > 30 s", "empty text"]


def test_empty_split_raises(tmp_path):
    import pytest

    from app.core.prepare import PrepareError

    with pytest.raises(PrepareError):
        build_manifest(None, None, probe=ok)({"train": [arow(" ")]}, tmp_path)


def test_spec_includes_tokenizer_identity():
    class C:
        base_model = "m"

    assert manifest_spec(C, "aa").params["tokenizer_sha16"] == "aa"
    assert manifest_spec(C, "aa").params["storage"] == "audio-ref"
    assert manifest_spec(C, "aa").model_dump() != manifest_spec(C, "bb").model_dump()
