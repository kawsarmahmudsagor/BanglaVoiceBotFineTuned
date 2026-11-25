"""qwen_asr phase module: light plumbing tests, a CPU test on a tiny random model, and the heavy VM test."""
import json
from pathlib import Path

import numpy as np
import pytest

from app.core.contract import read_config
from mlops_backend import qwen_asr as q

MODEL = "SayedShaun/Qwen3-ASR-0.6B-Bengali"
BN = ["আমি ভাত খাই", "তুমি কোথায় যাও", "সে বাড়ি ফিরেছে", "আমরা কাল আসব"]


def test_plan_init():
    assert q.plan_init(None) == "base"
    assert q.plan_init({"format": "hf-full"}) == "full_weights"
    with pytest.raises(ValueError, match="only hf-full"):
        q.plan_init({"format": "hf-lora"})


def test_ckpt_meta(tmp_path, run_dir):
    rd = run_dir("asr", MODEL)
    q.write_ckpt_meta(tmp_path, read_config(rd), 20)
    meta = json.loads((tmp_path / "init_meta.json").read_text())
    assert meta == {"backend": "qwen-asr", "base_model": MODEL, "method": "full", "format": "hf-full", "step": 20}


def test_start_path_uses_init_dir(run_dir):
    from app.core.backend_kit import write_init_meta
    from app.core.contract import InitRef

    rd = run_dir("asr", MODEL)
    cfg = read_config(rd)
    assert q.start_path(cfg, rd) == MODEL
    rd.init_dir.mkdir(parents=True, exist_ok=True)
    write_init_meta(rd.init_dir, backend="qwen-asr", base_model=MODEL, method="full", format="hf-full", step=5)
    cfg = cfg.model_copy(update={"init_from": InitRef(path=str(rd.init_dir), source_run="r0", checkpoint="final",
                                                      method="full", format="hf-full")})
    assert q.start_path(cfg, rd) == str(rd.init_dir)


def test_trainer_args_defaults_and_overrides(run_dir):
    rd = run_dir("asr", MODEL, {"max_steps": 100})
    args = q.qwen_trainer_args(read_config(rd), rd)
    assert args["learning_rate"] == 2e-5
    assert args["per_device_train_batch_size"] == 8 and args["gradient_accumulation_steps"] == 1
    assert args["warmup_steps"] == 0.02 and isinstance(args["warmup_steps"], float)  # ratio (transformers 5)
    assert args["bf16"] is True and args["activation_checkpointing"] is True
    assert args["max_steps"] == 100 and args["label_names"] == ["labels"]
    assert args["remove_unused_columns"] is False

    rd = run_dir("asr", MODEL, {"max_steps": 100, "save_steps": 25, "learning_rate": 1e-4, "batch_size": 2,
                                "grad_accum": 4, "warmup_ratio": 0.1, "gradient_checkpointing": False})
    args = q.qwen_trainer_args(read_config(rd), rd)
    assert args["save_steps"] == args["eval_steps"] == 25
    assert args["learning_rate"] == 1e-4 and args["per_device_train_batch_size"] == 2
    assert args["gradient_accumulation_steps"] == 4 and args["warmup_steps"] == 0.1
    assert args["activation_checkpointing"] is False


def test_with_defaults_keeps_recipe_values(run_dir):
    rd = run_dir("asr", MODEL, {"batch_size": 4})
    hp = q.with_defaults(read_config(rd)).hparams
    assert hp["batch_size"] == 4 and hp["learning_rate"] == 2e-5 and hp["max_label_tokens"] == 448


# ---------------------------------------------------------------- tiny random model (CPU, real processor)

@pytest.fixture(scope="module")
def real_processor():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{MODEL} processor not in the local HF cache ({type(e).__name__})")


def tiny_model():
    from transformers import AutoConfig, Qwen3ASRForConditionalGeneration

    cfg = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    a, t = cfg.audio_config, cfg.text_config
    a.d_model, a.encoder_layers, a.encoder_attention_heads, a.encoder_ffn_dim = 32, 1, 2, 64
    a.num_key_value_heads = 2
    a.downsample_hidden_size, a.output_dim = 8, 32
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 32, 64, 1
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 2, 1, 16
    t.layer_types = (t.layer_types or [])[:1] or None  # must match num_hidden_layers when the config is reloaded
    return Qwen3ASRForConditionalGeneration(cfg).float()  # from_config keeps the sub-configs' bf16 dtype


def test_tiny_model_loss_and_freeze(real_processor):
    """The collator's batch is accepted by the real model class; loss is finite; freezing the audio tower
    leaves the decoder and projector trainable."""
    import torch

    from mlops_backend.prompt import QwenAsrCollator

    torch.manual_seed(0)
    model = tiny_model()
    lens = {"a": 1.0, "b": 2.0}
    col = QwenAsrCollator(real_processor, loader=lambda p: (0.1 * np.random.default_rng(0).standard_normal(
        int(16000 * lens[p]))).astype("float32"))
    batch = col([{"audio_path": "a", "text": BN[0]}, {"audio_path": "b", "text": BN[1]}])
    out = model(**batch)
    assert torch.isfinite(out.loss)

    frozen = q.freeze_audio_encoder(model)
    assert frozen > 0
    assert not any(p.requires_grad for p in model.model.audio_tower.parameters())
    assert all(p.requires_grad for p in model.model.multi_modal_projector.parameters())
    assert all(p.requires_grad for p in model.model.language_model.parameters())
    out = model(**batch)
    out.loss.backward()
    assert all(p.grad is None for p in model.model.audio_tower.parameters())
    assert any(p.grad is not None for p in model.model.multi_modal_projector.parameters())


def test_tiny_model_transcriber_shape(real_processor):
    import torch

    torch.manual_seed(0)
    model = tiny_model().eval()
    tr = q.make_transcriber(model, real_processor, loader=lambda p: np.zeros(int(16000 * float(p)), "float32"))
    out = tr(["0.5", "1.0"])
    assert len(out) == 2 and all(isinstance(t, str) for t in out)


# ---------------------------------------------------------------- heavy (GPU VM)

@pytest.mark.heavy
def test_qwen_asr_20_steps(run_dir):
    """GPU VM only: prepare, 20 steps of full fine-tuning, eval with baseline, safetensors export that reloads
    and transcribes one clip."""
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    from app.core.backend_kit import Reporter
    from app.core.contract import read_eval_results, read_prepared_refs

    rd = run_dir("asr", MODEL, {"max_steps": 20, "batch_size": 2, "grad_accum": 1, "learning_rate": 1e-5,
                                "eval": {"max_samples": 4}}, texts=BN, seconds=1.5,
                 method={"type": "full"}, backend="qwen-asr", export=["safetensors"])
    cfg = read_config(rd)
    rep = Reporter(rd)
    assert q.prepare(rd, cfg, rep) == 0
    assert read_prepared_refs(rd)[0]["role"] == "train"
    assert q.train(rd, cfg, rep) == 0
    final = rd.checkpoints_dir / "final"
    assert (final / ".complete").exists()
    assert json.loads((final / "init_meta.json").read_text())["format"] == "hf-full"
    assert q.eval(rd, cfg, rep) == 0
    res = read_eval_results(rd).metrics
    assert {"wer", "cer", "rtf", "base/wer", "base/cer"} <= set(res)
    assert q.export(rd, cfg, rep) == 0
    out = rd.export_dir / "safetensors"
    assert list(out.glob("*.safetensors")) and rd.export_manifest_path.exists()

    import torch

    model = Qwen3ASRForConditionalGeneration.from_pretrained(str(out), dtype=torch.bfloat16).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(str(out))
    clip = next(iter(read_prepared_refs(rd)))["path"]
    wav = sorted((Path(clip) / "audio" / "train").glob("*.wav"))[0]
    texts = q.make_transcriber(model, processor)([str(wav)])
    assert len(texts) == 1 and isinstance(texts[0], str)
