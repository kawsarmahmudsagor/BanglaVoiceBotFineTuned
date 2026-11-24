import json

import pytest

from app.core.contract import read_config

PARLER = "ai4bharat/indic-parler-tts"
DEFAULT = "{speaker} speaks clearly in a studio-quality recording."


def test_describe_template():
    from mlops_backend.prepare.parler import describe

    assert describe({"speaker": "rina", "style": None}, DEFAULT) == "rina speaks clearly in a studio-quality recording."


def test_describe_style_and_collapse():
    from mlops_backend.prepare.parler import describe

    assert describe({"speaker": "rina", "style": "calm"}, "{speaker}   is {style}  here") == "rina is calm here"
    assert describe({"speaker": "rina"}, "{speaker} says {style}.") == "rina says ."


def tok(text):
    return [len(text), 1]


def test_row_to_example_ok():
    import numpy as np

    from mlops_backend.prepare.parler import row_to_example

    ex = row_to_example({"sample_id": "a", "text": "আমি  যাব", "speaker": "rina"}, np.zeros(44100, "float32"),
                        DEFAULT, tok, tok)
    assert ex["sample_id"] == "a" and ex["prompt_input_ids"] and ex["input_ids"]


def test_row_to_example_undecodable():
    from mlops_backend.prepare.parler import row_to_example

    assert row_to_example({"text": "x"}, RuntimeError("boom"), DEFAULT, tok, tok) == \
        "undecodable audio: RuntimeError"


def test_row_to_example_long_and_empty():
    import numpy as np

    from mlops_backend.prepare.parler import row_to_example

    long_audio = np.zeros(44100 * 31, "float32")
    assert row_to_example({"text": "x"}, long_audio, DEFAULT, tok, tok) == "duration 31.0 s > 30 s"
    assert row_to_example({"text": "   "}, np.zeros(10, "float32"), DEFAULT, tok, tok) == "empty text"


def test_loader_failure_becomes_exception():
    from mlops_backend.prepare.parler import _load

    def bad(path):
        raise ValueError("x")

    assert isinstance(_load(bad, "p"), ValueError)


def test_pad_code_labels_appends_eos_and_pads():
    from mlops_backend.prepare.parler import pad_code_labels

    out = pad_code_labels([([1, 2, 3, 4], 2), ([5, 6], 2)], eos_id=9)
    assert out[0] == [[1, 2], [3, 4], [9, 9]]
    assert out[1] == [[5, 6], [9, 9], [-100, -100]]


def test_spec_params(run_dir):
    from mlops_backend.prepare.parler import parler_spec

    cfg = read_config(run_dir("tts", PARLER))
    p = parler_spec(cfg, DEFAULT).params
    assert p["sample_rate"] == 44100 and p["description_template"] == DEFAULT and p["base_model"] == PARLER
    assert "bn_normalize_version" in p


def test_dispatch_parler_family():
    from mlops_backend import parler, phases

    assert phases._FAMILY_MODULE["parler"] == "parler"
    assert parler.prepare is parler.prepare_parler
    assert parler.train is parler.train_parler and parler.eval is parler.eval_parler
    assert parler.export is parler.export_parler


def test_init_plan():
    from mlops_backend.parler import plan_parler_init

    assert plan_parler_init(None) == "base"
    assert plan_parler_init({"format": "hf-full"}) == "full_weights"
    with pytest.raises(ValueError, match="only hf-full"):
        plan_parler_init({"format": "hf-lora"})


def test_ckpt_meta(run_dir, tmp_path):
    from mlops_backend.parler import write_ckpt_meta

    cfg = read_config(run_dir("tts", PARLER))
    d = tmp_path / "ck"
    d.mkdir()
    write_ckpt_meta(d, cfg, 3)
    meta = json.loads((d / "init_meta.json").read_text())
    assert meta["format"] == "hf-full" and meta["method"] == "full" and meta["step"] == 3


def test_template_from_hparams(run_dir):
    from mlops_backend.parler import template_of

    assert template_of(read_config(run_dir("tts", PARLER))) == DEFAULT
    assert template_of(read_config(run_dir("tts", PARLER, {"description_template": "{speaker}!"}))) == "{speaker}!"


def test_exporters_and_final_required(run_dir):
    from mlops_backend.parler import parler_exporters

    rd = run_dir("tts", PARLER)
    ex = parler_exporters(rd, read_config(rd))
    assert set(ex) == {"safetensors"}
    (rd.checkpoints_dir / "final").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no completed final checkpoint"):
        ex["safetensors"](rd.export_dir / "safetensors")


def test_eval_requires_complete_final(run_dir):
    from mlops_backend.parler import eval_parler

    rd = run_dir("tts", PARLER)
    (rd.checkpoints_dir / "final").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no completed final checkpoint"):
        eval_parler(rd, read_config(rd), None)


def test_trainer_args(run_dir):
    from mlops_backend.parler import parler_trainer_args

    rd = run_dir("tts", PARLER, {"max_steps": 20, "batch_size": 1})
    rd.prepared_refs_path.write_text('[{"role": "train", "path": "/p", "counts": {"train": 8, "val": 2}}]')
    a = parler_trainer_args(read_config(rd), rd)
    assert a["max_steps"] == 20 and a["label_names"] == ["labels"] and a["remove_unused_columns"] is False
    assert a["prediction_loss_only"] is True


@pytest.mark.heavy
def test_parler_20_steps_and_synthesize(run_dir, monkeypatch, tmp_path):
    """GPU VM only: prepare tiny_audio_ds-style rows, 20 steps, eval (loss + rtf), safetensors export."""
    from app.core.backend_kit import Reporter
    from app.core.contract import read_eval_results
    from mlops_backend import parler as p

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    rd = run_dir("tts", PARLER, {"max_steps": 20, "batch_size": 1, "grad_accum": 1, "learning_rate": 1e-5},
                 export=["safetensors"])
    cfg = read_config(rd)
    rep = Reporter(rd)
    assert p.prepare(rd, cfg, rep) == 0
    assert p.train(rd, cfg, rep) == 0
    final = rd.checkpoints_dir / "final"
    assert (final / ".complete").exists()
    assert json.loads((final / "init_meta.json").read_text())["format"] == "hf-full"
    assert p.eval(rd, cfg, rep) == 0
    res = read_eval_results(rd).metrics
    assert "eval_loss" in res and "rtf" in res
    assert p.export(rd, cfg, rep) == 0
    out = rd.export_dir / "safetensors"
    assert list(out.glob("*.safetensors")) and (out / "config.json").exists()
    assert (out / "description_tokenizer").is_dir()
    assert (out / "preprocessor_config.json").exists()
    assert rd.export_manifest_path.exists()


def test_hop_derivation():
    from types import SimpleNamespace as NS

    from mlops_backend.prepare.parler import _hop

    assert _hop(NS(config=NS(sampling_rate=44100, frame_rate=86))) == 44100 / 86
    assert _hop(NS(config=NS(upsampling_ratios=[8, 5, 4, 2]))) == 320
    assert _hop(NS(config=NS())) == 512


def test_trim_codes_per_row_frames():
    import numpy as np

    from mlops_backend.prepare.parler import trim_codes

    codes = np.arange(2 * 3 * 10).reshape(2, 3, 10)
    out = trim_codes(codes, [1000, 2500], 512)
    assert [o.shape for o in out] == [(3, 2), (3, 5)] and out[0].dtype == np.int16


def test_iter_records_pipeline_with_fakes():
    import numpy as np

    from mlops_backend.prepare.parler import iter_records

    def loader(path):
        if path == "bad":
            raise OSError("x")
        return np.zeros(44100, "float32")

    def enc(audios):
        return [np.ones((3, 4), np.int16) for _ in audios]

    rows = [{"sample_id": "a", "audio_path": "ok", "text": "আমি", "speaker": "r"},
            {"sample_id": "b", "audio_path": "bad", "text": "আমি"}]
    skipped, ticks = [], []
    recs = list(iter_records(rows, "train", skipped, enc, loader, DEFAULT, tok, tok, ticks.append))
    assert len(recs) == 1 and recs[0]["num_codebooks"] == 3 and len(recs[0]["codes"]) == 12
    assert skipped[0].reason == "undecodable audio: OSError" and skipped[0].sample_id == "b"
    assert sum(ticks) == 2


def test_baseline_key_has_template(run_dir):
    from mlops_backend.parler import parler_baseline_key

    a = parler_baseline_key(read_config(run_dir("tts", PARLER)), 5)
    b = parler_baseline_key(read_config(run_dir("tts", PARLER, {"description_template": "{speaker}!"})), 5)
    assert a != b and a["max_samples"] == 5


@pytest.mark.heavy
def test_write_split_roundtrip(tmp_path):
    import numpy as np
    from datasets import load_from_disk

    from mlops_backend.prepare.parler import write_split

    recs = [{"sample_id": str(i), "prompt_input_ids": [1], "input_ids": [2],
             "codes": np.ones(6, np.int16), "num_codebooks": 3} for i in range(5)]
    assert write_split(iter(recs), tmp_path / "train", shard_rows=2) == 5
    assert len(load_from_disk(str(tmp_path / "train"))) == 5
    assert not (tmp_path / "train.shards").exists()


def test_parler_is_gated():
    from mlops_backend.models import MODELS

    assert MODELS[PARLER].gated is True
