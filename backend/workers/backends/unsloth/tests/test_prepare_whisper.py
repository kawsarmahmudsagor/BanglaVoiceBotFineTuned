"""Heavy (VM): real WhisperProcessor and datasets."""
import json

import pytest

pytestmark = pytest.mark.heavy

WHISPER = "openai/whisper-large-v3-turbo"


def _build(tmp_path, rows):
    from transformers import WhisperProcessor

    from app.core.backend_kit.whisper import build_whisper
    proc = WhisperProcessor.from_pretrained(WHISPER)
    out = tmp_path / "out"
    out.mkdir()
    return build_whisper(proc)(rows, out), out, proc


def test_whisper_build_skips_long_audio(tmp_path):
    import numpy as np

    from app.core.backend_kit.audio import write_wav
    sr = 16000
    t = np.arange(35 * sr) / sr
    write_wav(tmp_path / "long.wav", (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr)
    write_wav(tmp_path / "short.wav", (0.3 * np.sin(2 * np.pi * 220 * t[:sr])).astype("float32"), sr)
    rows = {"train": [{"sample_id": "long", "audio_path": str(tmp_path / "long.wav"), "text": "আমি"},
                      {"sample_id": "short", "audio_path": str(tmp_path / "short.wav"), "text": "আমি"}]}
    res, out, proc = _build(tmp_path, rows)
    assert res.counts == {"train": 1}
    assert [(s.sample_id, s.reason) for s in res.skipped] == [("long", "duration 35.0 s > 30 s")]
    from datasets import load_from_disk
    ds = load_from_disk(str(out / "train"))
    assert len(ds) == 1
    assert ds[0]["audio_path"].endswith("short.wav") and "audio" not in ds.column_names
    from app.core.backend_kit.whisper import WhisperCollator
    batch = WhisperCollator(proc)([ds[0]])
    assert batch["input_features"].shape[1] == proc.feature_extractor.feature_size


def test_whisper_lora_20_steps(run_dir, tiny_audio_ds):
    from app.core.backend_kit import Reporter
    from app.core.contract import read_config
    from mlops_backend.asr import eval_asr, train_asr

    root, names = tiny_audio_ds(n=4)
    rd = run_dir("asr", WHISPER, {"max_steps": 20, "batch_size": 1, "grad_accum": 1,
                                  "eval": {"max_samples": 2}})
    cfg = read_config(rd)
    from mlops_backend.asr import prepare as prepare_asr
    rep = Reporter(rd)
    assert prepare_asr(rd, cfg, rep) == 0
    assert train_asr(rd, cfg, rep) == 0
    assert (rd.checkpoints_dir / "final" / ".complete").exists()
    meta = json.loads((rd.checkpoints_dir / "final" / "init_meta.json").read_text())
    assert meta["format"] == "hf-peft"
    assert eval_asr(rd, cfg, rep) == 0
    from app.core.contract import read_eval_results
    m = read_eval_results(rd).metrics
    for k in ("eval_loss", "wer", "base/wer"):
        assert k in m
