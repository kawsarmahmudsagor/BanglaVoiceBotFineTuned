import json

import pytest

from mlops_backend.prepare.ctc import (
    build_ctc, build_ctc_vocab, ctc_greedy_decode, ctc_labels, ctc_spec, pad_ctc_labels, vocab_hash,
)

V = {"আ": 0, "|": 1, "[UNK]": 2, "[PAD]": 3}


def row(text, path="a.wav", sid="s"):
    return {"sample_id": sid, "audio_path": path, "text": text, "duration_s": 1.0}


def ok(path):
    return 1.0


def test_vocab_maps_space_to_pipe_and_adds_specials():
    v = build_ctc_vocab(["আমি যাব"])
    assert "|" in v and " " not in v and list(v)[-2:] == ["[UNK]", "[PAD]"]


def test_vocab_is_deterministic():
    assert build_ctc_vocab(["খা", "আ"]) == build_ctc_vocab(["আ", "খা"])


def test_vocab_accepts_a_generator():
    assert "আ" in build_ctc_vocab(t for t in ["আ"])


def test_mms_skips_out_of_vocab_rows(tmp_path):
    res = build_ctc(V, "fixed", probe=ok)({"train": [row("আ আ"), row("আক", sid="bad")]}, tmp_path)
    assert res.counts["train"] == 1 and res.skipped[0].reason == "characters not in vocab: ক"
    assert res.skipped[0].sample_id == "bad"


def test_new_mode_never_skips_for_vocab(tmp_path):
    res = build_ctc(V, "new", probe=ok)({"train": [row("আক")]}, tmp_path)
    assert res.counts["train"] == 1 and not res.skipped


def test_build_writes_jsonl_refs_and_vocab(tmp_path):
    build_ctc(V, "fixed", probe=ok)({"train": [row("আ  আ", "/x/a.wav")]}, tmp_path)
    rec = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rec == [{"audio_path": "/x/a.wav", "duration_s": 1.0, "text": "আ আ"}]
    assert json.loads((tmp_path / "vocab.json").read_text(encoding="utf-8")) == V


def test_long_audio_skipped(tmp_path):
    long_row = row("আ", sid="long") | {"duration_s": 31.0}
    res = build_ctc(V, "new", probe=ok)({"train": [row("আ"), long_row]}, tmp_path)
    assert res.skipped[0].reason == "duration 31.0 s > 30 s"


def test_undecodable_audio_skipped(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    from app.core.backend_kit.audio import write_wav
    import numpy as np

    write_wav(tmp_path / "good.wav", np.zeros(1600, dtype="float32"), 16000)
    rows = {"train": [{"sample_id": "g", "audio_path": str(tmp_path / "good.wav"), "text": "আ"},
                      {"sample_id": "b", "audio_path": str(bad), "text": "আ"}]}
    out = tmp_path / "out"
    res = build_ctc(V, "new")(rows, out)  # default probe: real decode check
    assert res.counts == {"train": 1}
    assert res.skipped[0].sample_id == "b" and res.skipped[0].reason.startswith("undecodable audio: ")


def test_empty_split_raises(tmp_path):
    from app.core.prepare import PrepareError

    with pytest.raises(PrepareError, match="split train is empty"):
        build_ctc(V, "fixed", probe=ok)({"train": [row("ক")]}, tmp_path)


def test_progress_reported(tmp_path):
    seen = []

    class Rep:
        def status(self, phase, step, total, msg):
            seen.append((step, total))

    build_ctc(V, "new", Rep(), probe=ok)({"train": [row("আ")] * 3}, tmp_path)
    assert seen[-1] == (3, 3)


def test_spec_contains_vocab_hash(run_dir):
    from app.core.contract import read_config

    cfg = read_config(run_dir("asr", "facebook/wav2vec2-xls-r-300m"))
    a, b = ctc_spec(cfg, V), ctc_spec(cfg, {**V, "ক": 4})
    assert a.params["vocab_sha16"] == vocab_hash(V) and a.hash() != b.hash()


def test_labels_and_padding():
    assert ctc_labels("আ আক", V) == [0, 1, 0, 2]
    assert pad_ctc_labels([[1], [1, 2]]) == [[1, -100], [1, 2]]


def test_greedy_decode_collapses_and_drops_blank():
    inv = {0: "আ", 1: "|", 2: "ক", 3: ""}
    assert ctc_greedy_decode([0, 0, 3, 0, 1, 2, 2, 3], inv, pad_id=3) == "আআ ক"


def test_prepare_ctc_cold_end_to_end(run_dir):
    from app.core.contract import read_config, read_prepared_refs
    from mlops_backend.ctc import prepare_ctc

    rd = run_dir("asr", "facebook/wav2vec2-xls-r-300m")
    assert prepare_ctc(rd, read_config(rd), None) == 0
    refs = read_prepared_refs(rd)
    assert {r["role"] for r in refs} == {"train", "eval"}
    train = next(r for r in refs if r["role"] == "train")
    from pathlib import Path

    assert (Path(train["path"]) / "vocab.json").exists() and (Path(train["path"]) / "train.jsonl").exists()
    assert not list(Path(train["path"]).glob("*.wav"))


def test_greedy_decode_drops_unk():
    inv = {0: "আ", 1: "[UNK]", 2: "[PAD]"}
    assert ctc_greedy_decode([0, 1, 0, 2, 1], inv, pad_id=2, unk_id=1) == "আআ"


def test_empty_text_skipped(tmp_path):
    res = build_ctc(V, "new", probe=ok)({"train": [row("আ"), row("  ", sid="e")]}, tmp_path)
    assert res.counts["train"] == 1 and res.skipped[0].reason == "empty text"


def test_first_last_lines(tmp_path):
    from mlops_backend.prepare.ctc import first_last_lines

    p = tmp_path / "t.jsonl"
    p.write_text("".join(json.dumps({"i": i, "pad": "x" * 5000}) + "\n" for i in range(5)), encoding="utf-8")
    assert [r["i"] for r in first_last_lines(p)] == [0, 4]
    p.write_text(json.dumps({"i": 9}) + "\n")
    assert [r["i"] for r in first_last_lines(p)] == [9, 9]
    p.write_text("")
    assert first_last_lines(p) == []
