"""Qwen3-ASR converter (light: char-level fake tokenizer, no torch/transformers)."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.prepare import PrepareError
from app.core.text import BN_NORMALIZE_VERSION, bn_normalize

from mlops_backend.prepare.qwen_asr import (
    MAX_LABEL_TOKENS, build_qwen_asr, check_prepared, convert_clip, label_token_count, qwen_asr_spec, read_rows,
    tokenizer_hash,
)
from mlops_backend.prompt import PREFIX


def chars(text):
    return [ord(c) for c in text]


def row(sid, path, text="আমি  যাব।", dur=None):
    return {"sample_id": sid, "audio_path": str(path), "text": text, "duration_s": dur}


def test_spec_params():
    spec = qwen_asr_spec(SimpleNamespace(base_model="SayedShaun/Qwen3-ASR-0.6B-Bengali"), "abc123", 300)
    assert spec.family == "qwen-asr"
    p = spec.params
    assert p["base_model"] == "SayedShaun/Qwen3-ASR-0.6B-Bengali"
    assert p["tokenizer_sha16"] == "abc123"
    assert p["prompt_prefix"] == PREFIX
    assert p["bn_normalize_version"] == BN_NORMALIZE_VERSION
    assert p["max_label_tokens"] == 300 and p["sample_rate"] == 16000 and p["max_seconds"] == 30
    assert MAX_LABEL_TOKENS == 448


def test_tokenizer_hash_tracks_vocab():
    a = SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 1})
    b = SimpleNamespace(get_vocab=lambda: {"b": 1, "a": 0})
    c = SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 1, "<asr_text>": 2})
    assert tokenizer_hash(a) == tokenizer_hash(b) != tokenizer_hash(c)
    assert len(tokenizer_hash(a)) == 16


def test_label_count_includes_eos():
    assert label_token_count("abc", chars) == 4


def test_build_writes_16k_mono_wavs_and_jsonl(tiny_tts_ds, tmp_path):
    import soundfile as sf

    _, rows = tiny_tts_ds(4)  # 22.05 / 44.1 kHz sources
    build = build_qwen_asr(chars)
    out = tmp_path / "out"
    res = build({"train": rows[:3], "val": rows[3:]}, out)
    assert res.counts == {"train": 3, "val": 1} and res.skipped == []
    assert set(res.files) == {"train.jsonl", "val.jsonl", "audio"}
    lines = [json.loads(x) for x in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [set(r) for r in lines] == [{"audio", "text", "duration", "sample_id"}] * 3
    for r, src in zip(lines, rows):
        assert not r["audio"].startswith("/") and r["audio"].startswith("audio/train/")
        assert r["text"] == bn_normalize(src["text"], mode="asr")
        info = sf.info(str(out / r["audio"]))
        assert info.samplerate == 16000 and info.channels == 1
        assert r["duration"] == pytest.approx(0.5, abs=0.01)
    assert [r["sample_id"] for r in lines] == ["t0", "t1", "t2"]


def test_skips_long_empty_tokens_and_undecodable(tmp_path):
    calls = []

    def loader(path):
        calls.append(path)
        if path.endswith("bad.wav"):
            raise RuntimeError("cannot decode")
        secs = 31.0 if path.endswith("long.wav") else 1.0
        return np.zeros(int(16000 * secs), dtype="float32")

    written = []
    build = build_qwen_asr(chars, max_label_tokens=10, loader=loader, writer=lambda p, a: written.append(p))
    rows = [row("ok", "/d/ok.wav", "আমি যাব"),
            row("given-long", "/d/given.wav", dur=40.0),
            row("decoded-long", "/d/long.wav"),
            row("empty", "/d/e.wav", text="  "),
            row("tokens", "/d/t.wav", text="অনেক বড় একটা বাক্য"),
            row("bad", "/d/bad.wav")]
    res = build({"train": rows}, tmp_path / "out")
    assert res.counts == {"train": 1}
    reasons = {s.sample_id: s.reason for s in res.skipped}
    assert reasons["given-long"] == "duration 40.0 s > 30 s"
    assert reasons["decoded-long"] == "duration 31.0 s > 30 s"
    assert reasons["empty"] == "empty text"
    assert reasons["tokens"].endswith("label tokens > 10")
    assert reasons["bad"] == "undecodable audio: RuntimeError"
    assert "/d/given.wav" not in calls and "/d/t.wav" not in calls and "/d/e.wav" not in calls  # never decoded
    assert len(written) == 1


def test_convert_clip_returns_duration(tmp_path):
    got = []
    res = convert_clip("x", tmp_path / "a.wav", None, loader=lambda p: np.zeros(8000, "float32"),
                       writer=lambda p, a: got.append(len(a)))
    assert res == 0.5 and got == [8000]


def test_empty_split_raises(tmp_path):
    build = build_qwen_asr(chars, loader=lambda p: np.zeros(16000, "float32"), writer=lambda p, a: None)
    with pytest.raises(PrepareError, match="split val is empty"):
        build({"train": [row("a", "/d/a.wav")], "val": [row("b", "/d/b.wav", text="")]}, tmp_path / "out")


def test_ensure_prepared_paths_survive_rename(run_dir, tmp_path):
    """The cache is built in a temp dir and renamed: rows must resolve against the final dir."""
    from app.core.contract import read_config
    from app.core.prepare import ensure_prepared

    rd = run_dir("asr", "SayedShaun/Qwen3-ASR-0.6B-Bengali", texts=["আমি", "তুমি", "সে", "আমরা"])
    cfg = read_config(rd)
    spec = qwen_asr_spec(cfg, "h" * 16)
    pi = ensure_prepared(cfg, cfg.datasets["train"], spec, build_qwen_asr(chars), splits=["train", "val"])
    rows = read_rows(pi.path, "train")
    assert rows and all(os.path.exists(r["audio_path"]) for r in rows)
    check_prepared(pi.path)
    Path(rows[0]["audio_path"]).unlink()
    with pytest.raises(RuntimeError, match="re-run prepare"):
        check_prepared(pi.path)
