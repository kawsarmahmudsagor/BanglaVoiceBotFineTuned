"""F5 converter (light: no torch / f5_tts / network)."""
import json

import numpy as np
import pytest

from app.core.text import bn_normalize

from mlops_backend.prepare.f5 import (
    Decoded, build_f5, extend_vocab, f5_spec, pick_vocab_file, read_vocab, vocab_hash,
)


def arow(text="আ", path="/x/a.wav", sid="s", dur=1.0, speaker="spk"):
    return {"sample_id": sid, "audio_path": path, "text": text, "duration_s": dur, "speaker": speaker}


def arrow_rows(p):
    import pyarrow as pa

    with pa.OSFile(str(p), "rb") as f:
        return pa.ipc.open_stream(f).read_all().to_pylist()


def test_f5_appends_new_chars_without_reordering():
    assert extend_vocab(["আ", "ম"], ["মাআক", "খ"]) == ["আ", "ম", "া", "ক", "খ"]


def test_build_references_source_and_writes_duration_json(tiny_tts_ds, tmp_path):
    import soundfile as sf

    _, rows = tiny_tts_ds(n=4)
    out = tmp_path / "out"
    res = build_f5([" ", "আ"])({"train": rows[:3], "val": rows[3:]}, out)
    assert res.counts == {"train": 3, "val": 1}
    tr = arrow_rows(out / "train" / "raw.arrow")
    assert [r["audio_path"] for r in tr] == [r["audio_path"] for r in rows[:3]]
    assert tr[0]["text"] == bn_normalize(rows[0]["text"], mode="tts") and tr[0]["duration"] == pytest.approx(0.5, abs=1e-3)
    dj = json.loads((out / "train" / "duration.json").read_text(encoding="utf-8"))
    assert dj == {"duration": [r["duration"] for r in tr]}
    vocab = read_vocab(out / "vocab.txt")
    assert vocab[:2] == [" ", "আ"] and "ম" in vocab and "শ" in vocab
    assert not (out / "wavs").exists()
    # one 24 kHz reference per speaker
    for spk in ("a", "b"):
        info = sf.info(str(out / "refs" / f"{spk}.wav"))
        assert info.samplerate == 24000 and info.channels == 1
        assert (out / "refs" / f"{spk}.txt").read_text(encoding="utf-8").startswith("আমি যাব")
    assert "train/raw.arrow" in res.files and "vocab.txt" in res.files


def test_unreadable_format_transcoded_to_flac(tmp_path):
    import soundfile as sf

    audio = np.zeros(12000, dtype="float32")
    loader = lambda p, sr: audio  # noqa: E731
    probe = lambda p: Decoded(audio)  # noqa: E731
    out = tmp_path / "o"
    res = build_f5([], probe=probe, loader=loader)({"train": [arow(path="/x/a.m4a", sid="sid1", dur=0)]}, out)
    flac = out / "wavs" / "sid1.flac"
    assert res.counts["train"] == 1 and flac.is_file()
    info = sf.info(str(flac))
    assert info.samplerate == 24000 and info.channels == 1
    rec = arrow_rows(out / "train" / "raw.arrow")[0]
    assert rec["audio_path"] == str(flac.resolve()) and rec["duration"] == pytest.approx(0.5)


def test_undecodable_audio_skipped(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    good = tmp_path / "g.wav"
    import soundfile as sf

    sf.write(str(good), np.zeros(2400, dtype="float32"), 24000)
    from mlops_backend.prepare import f5 as m

    probe = m.make_probe(lambda p, sr: (_ for _ in ()).throw(ValueError("x")))
    res = build_f5([], probe=probe)(
        {"train": [arow(path=str(bad), sid="b", dur=0), arow(path=str(good), sid="g", dur=0)]}, tmp_path / "o")
    assert res.counts["train"] == 1
    assert res.skipped[0].reason == "undecodable audio: ValueError"


def test_long_and_empty_skipped(tmp_path):
    probe = lambda p: 31.0 if p == "/x/long.wav" else 1.0  # noqa: E731
    build = build_f5([], probe=probe, loader=lambda p, sr: np.zeros(100, dtype="float32"))
    res = build({"train": [arow(), arow(path="/x/long.wav", sid="l"), arow("  ", sid="e")]}, tmp_path)
    assert [s.reason for s in res.skipped] == ["duration 31.0 s > 30 s", "empty text"]


def test_empty_split_raises(tmp_path):
    from app.core.prepare import PrepareError

    with pytest.raises(PrepareError):
        build_f5([], probe=lambda p: 1.0)({"train": [arow("  ")]}, tmp_path)


def test_f5_vocab_extends_init_vocab(run_dir, tmp_path):
    from app.core.contract import InitRef, write_config
    from mlops_backend.f5 import base_vocab

    rd = run_dir("tts", "ai4bharat/IndicF5")
    from app.core.contract import read_config

    cfg = read_config(rd)
    cfg = cfg.model_copy(update={"init_from": InitRef(path=str(rd.init_dir), source_run="r0", checkpoint="final",
                                                      method="full", format="f5")})
    write_config(rd, cfg)
    rd.init_dir.mkdir(parents=True, exist_ok=True)
    (rd.init_dir / "vocab.txt").write_bytes("a\n \nআ\n".encode("utf-8"))

    def boom():
        raise AssertionError("hub vocab must not be used on a warm start")

    assert base_vocab(rd, cfg, boom) == ["a", " ", "আ"]
    fresh = cfg.model_copy(update={"init_from": None})
    assert base_vocab(rd, fresh, lambda: ["z"]) == ["z"]
    assert f5_spec(cfg, vocab_hash(["a"])).params["base_vocab_sha16"] == vocab_hash(["a"])
    assert vocab_hash(["a"]) != vocab_hash(["a", "b"])


def test_pick_vocab_file():
    files = ["config.json", "model.safetensors", "checkpoints/vocab.txt", "vocab.txt", "tokenizer/vocab_old.txt"]
    assert pick_vocab_file(files) == "vocab.txt"
    assert pick_vocab_file(["a/b/vocab.txt", "x/vocab.txt"]) == "x/vocab.txt"
    with pytest.raises(Exception):
        pick_vocab_file(["config.json"])
