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
    res = build_manifest(None, None, probe=lambda p: 31.0 if p == "/x/long.wav" else 1.0)(
        {"train": [arow("আ"), arow("আ", "/x/long.wav", sid="l", dur=31.0), arow("  ", sid="e")]}, tmp_path)
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


def test_duration_is_the_probed_file_duration(tmp_path):
    res = build_manifest(None, None, probe=lambda p: 2.5)({"train": [arow("আ", dur=9.0)]}, tmp_path)
    assert res.counts["train"] == 1
    assert lines(tmp_path / "train.jsonl")[0]["duration"] == 2.5


def test_duration_falls_back_to_row_when_probe_empty(tmp_path):
    build_manifest(None, None, probe=lambda p: 0)({"train": [arow("আ", dur=3.0)]}, tmp_path)
    assert lines(tmp_path / "train.jsonl")[0]["duration"] == 3.0


def test_audio_filepath_is_resolved(tmp_path):
    build_manifest(None, None, probe=ok)({"train": [arow("আ", str(tmp_path / "x" / ".." / "a.wav"))]}, tmp_path / "o")
    from pathlib import Path

    assert lines(tmp_path / "o" / "train.jsonl")[0]["audio_filepath"] == str((tmp_path / "a.wav").resolve())
    assert Path(lines(tmp_path / "o" / "train.jsonl")[0]["audio_filepath"]).is_absolute()


class FakeTok:
    vocab = ["<unk>", "আ", "ক"]
    unk_id = None

    def token_to_id(self, t):
        return self.vocab.index(t) if t in self.vocab else None


def test_unk_id_derived_from_token():
    from mlops_backend.asr import resolve_unk_id

    assert resolve_unk_id(FakeTok(), "m") == 0


def test_unk_id_missing_raises():
    import pytest

    from app.core.prepare import PrepareError
    from mlops_backend.asr import resolve_unk_id

    class T:
        unk_id = None

    with pytest.raises(PrepareError, match="tokenizer of m has no unk id; cannot guard unknown characters"):
        resolve_unk_id(T(), "m")


def test_tokenizer_identity_follows_vocab():
    from mlops_backend.asr import tokenizer_identity

    class A:
        vocab = ["a", "b"]

    class B:
        vocab = ["a", "c"]

    assert tokenizer_identity(A()) == tokenizer_identity(A()) != tokenizer_identity(B())


def test_cache_hit_checks_audio_refs(tmp_path):
    import pytest

    from mlops_backend.asr import check_manifest_refs

    (tmp_path / "train.jsonl").write_text(
        json.dumps({"audio_filepath": str(tmp_path / "gone.wav"), "duration": 1, "text": "আ"}) + "\n",
        encoding="utf-8")
    info = type("I", (), {"path": str(tmp_path), "files": ["train.jsonl"]})()
    with pytest.raises(RuntimeError, match="missing audio"):
        check_manifest_refs(info)
    (tmp_path / "gone.wav").write_bytes(b"x")
    check_manifest_refs(info)
