import json
import sys

import pytest

from app.core.contract import InitRef, read_config

WHISPER = "openai/whisper-large-v3-turbo"


def test_dispatch_whisper_family(tmp_path, monkeypatch):
    from mlops_backend import phases, whisper_full

    assert phases._FAMILY_MODULE["whisper"] == "whisper_full"
    seen = {}
    for phase in ("prepare", "train", "eval", "export"):
        fn = getattr(whisper_full, phase)
        monkeypatch.setattr(whisper_full, phase, lambda rd, cfg, rep, p=phase: seen.setdefault(p, 0))
    from app.core.contract import Phase, RunConfig, RunDir

    rd = RunDir(tmp_path / "run")
    rd.ensure()
    cfg = RunConfig(run_id="r1", attempt=1, resume=False, task="asr", backend="hf", base_model=WHISPER,
                    method={"type": "full"}, hparams={}, export=[], seed=1, datasets={}, prepared_root="p")
    for p in Phase:
        if p.value in ("prepare", "train", "eval", "export"):
            assert phases._dispatch(p)(rd, cfg, None) == 0
    assert set(seen) == {"prepare", "train", "eval", "export"}


def test_handler_names():
    from app.core.backend_kit.whisper import prepare_whisper
    from mlops_backend import whisper_full as w

    assert w.prepare is prepare_whisper
    assert w.train is w.train_whisper and w.eval is w.eval_whisper and w.export is w.export_whisper


def test_exporters_safetensors_and_ct2(run_dir):
    from mlops_backend.whisper_full import whisper_exporters

    rd = run_dir("asr", WHISPER)
    assert set(whisper_exporters(rd, read_config(rd))) == {"safetensors", "ct2"}


def test_exporter_requires_complete_final(run_dir):
    from mlops_backend.whisper_full import whisper_exporters

    rd = run_dir("asr", WHISPER)
    (rd.checkpoints_dir / "final").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no completed final checkpoint"):
        whisper_exporters(rd, read_config(rd))["safetensors"](rd.export_dir / "safetensors")


def test_eval_requires_complete_final(run_dir):
    from mlops_backend.whisper_full import eval_whisper

    rd = run_dir("asr", WHISPER)
    (rd.checkpoints_dir / "final").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no completed final checkpoint"):
        eval_whisper(rd, read_config(rd), None)


def test_trainer_args(run_dir):
    from mlops_backend.whisper_full import whisper_trainer_args

    rd = run_dir("asr", WHISPER, {"max_steps": 20, "batch_size": 1})
    rd.prepared_refs_path.write_text('[{"role": "train", "path": "/p", "counts": {"train": 8, "val": 2}}]')
    a = whisper_trainer_args(read_config(rd), rd)
    assert a["max_steps"] == 20 and a["per_device_train_batch_size"] == 1
    assert a["gradient_checkpointing"] is True and a["bf16"] is True
    assert a["remove_unused_columns"] is False and a["label_names"] == ["labels"]
    assert a["dataloader_num_workers"] >= 1 and a["optim"] == "adamw_torch"
    assert a["predict_with_generate"] is False


def test_init_plan_accepts_only_hf_full():
    from mlops_backend.whisper_full import plan_whisper_init

    assert plan_whisper_init(None) == "base"
    assert plan_whisper_init({"format": "hf-full"}) == "full_weights"
    with pytest.raises(ValueError, match="hf-peft.*only hf-full"):
        plan_whisper_init({"format": "hf-peft"})


def test_ckpt_meta_is_hf_full(run_dir, tmp_path):
    from mlops_backend.whisper_full import write_ckpt_meta

    cfg = read_config(run_dir("asr", WHISPER))
    d = tmp_path / "ck"
    d.mkdir()
    write_ckpt_meta(d, cfg, 7)
    meta = json.loads((d / "init_meta.json").read_text())
    assert meta["format"] == "hf-full" and meta["backend"] == "hf" and meta["step"] == 7


def test_baseline_key_varies(run_dir):
    from app.core.backend_kit.whisper import asr_baseline_key

    cfg = read_config(run_dir("asr", WHISPER))
    a = asr_baseline_key(cfg, None)
    assert asr_baseline_key(cfg, 5) != a
    w = cfg.model_copy(update={"init_from": InitRef(path="p", source_run="r0", checkpoint="final",
                                                    method="full", format="hf-full")})
    assert asr_baseline_key(w, None)["init"] == ["r0", "final"]


def test_processor_kwarg_picks_supported_name():
    from app.core.backend_kit.whisper import processor_kwarg

    class New:
        def __init__(self, processing_class=None): ...

    class Old:
        def __init__(self, tokenizer=None): ...

    assert processor_kwarg(New, 1) == {"processing_class": 1}
    assert processor_kwarg(Old, 1) == {"tokenizer": 1}


def test_ct2_command_shared(tmp_path):
    from app.core.backend_kit.whisper import ct2_command

    cmd = ct2_command(tmp_path / "m", tmp_path / "o")
    assert cmd[0] == sys.executable or cmd[0].endswith("ct2-transformers-converter")
    assert cmd[cmd.index("--quantization") + 1] == "float16"


@pytest.mark.heavy
def test_whisper_full_20_steps(run_dir, monkeypatch, tmp_path):
    """GPU VM only: 20 steps of full fine-tuning, eval with baseline, both exports."""
    from app.core.backend_kit import Reporter
    from app.core.backend_kit.whisper import prepare_whisper
    from app.core.contract import read_eval_results
    from mlops_backend import whisper_full as w

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    rd = run_dir("asr", WHISPER, {"max_steps": 20, "batch_size": 1, "grad_accum": 1, "learning_rate": 1e-5},
                 export=["safetensors", "ct2"])
    cfg = read_config(rd)
    rep = Reporter(rd)
    assert prepare_whisper(rd, cfg, rep) == 0
    assert w.train(rd, cfg, rep) == 0
    final = rd.checkpoints_dir / "final"
    assert (final / ".complete").exists()
    assert json.loads((final / "init_meta.json").read_text())["format"] == "hf-full"
    assert w.eval(rd, cfg, rep) == 0
    res = read_eval_results(rd).metrics
    assert "base/wer" in res and "wer" in res
    assert w.export(rd, cfg, rep) == 0
    assert (rd.export_dir / "safetensors" / "model.safetensors").exists() or list(
        (rd.export_dir / "safetensors").glob("*.safetensors"))
    assert (rd.export_dir / "ct2" / "model.bin").exists()
    assert rd.export_manifest_path.exists()
