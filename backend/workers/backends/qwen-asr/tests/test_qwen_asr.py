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


def test_tiny_batched_generate_matches_single(real_processor):
    """Left-padded batched greedy generation gives the same tokens as one clip at a time (padding and
    position ids are handled), for clips of different lengths."""
    import torch

    from mlops_backend.prompt import generation_inputs, prompt_text

    torch.manual_seed(0)
    model = tiny_model().eval()
    rng = np.random.default_rng(1)
    audios = [(0.1 * rng.standard_normal(int(16000 * s))).astype("float32") for s in (0.7, 2.3, 1.4)]
    prompt = prompt_text(real_processor)

    def gen(batch):
        inputs = generation_inputs(real_processor, prompt, batch)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=6, do_sample=False)
        return out[:, inputs["input_ids"].shape[1]:].tolist()

    assert gen(audios) == [gen([a])[0] for a in audios]


class _Probe:
    """Records, at each training step, whether the model is gradient-checkpointed."""

    def __init__(self):
        from transformers import TrainerCallback

        seen = self.seen = []

        class P(TrainerCallback):
            def on_step_begin(self, args, state, control, model=None, **kw):
                seen.append(bool(getattr(model, "is_gradient_checkpointing", False)))

        self.callback = P()


def _wav_rows(root, durations):
    from app.core.backend_kit.audio import write_wav

    rows = []
    for i, d in enumerate(durations):
        path = root / f"c{i}.wav"
        t = np.arange(int(16000 * d)) / 16000
        write_wav(path, (0.1 * np.sin(2 * np.pi * 220 * t)).astype("float32"), 16000)
        rows.append({"audio_path": str(path), "text": BN[i % 4], "duration": d, "sample_id": f"c{i}"})
    return rows


@pytest.mark.parametrize("ckpt", [True, False])
def test_tiny_trainer_checkpointing_follows_hparam(real_processor, run_dir, tmp_path, ckpt):
    """With gradient_checkpointing true the model is really checkpointed while training; with false it is not."""
    import torch

    from app.core.backend_kit import StopFlag

    torch.manual_seed(0)
    rd = run_dir("asr", MODEL, {"max_steps": 2, "batch_size": 2, "gradient_checkpointing": ckpt})
    ds = q.RowsDataset(_wav_rows(tmp_path, [0.5, 1.0, 1.5, 2.0]))
    trainer = q.make_trainer(read_config(rd), rd, None, tiny_model(), real_processor, ds, ds.select([0, 1]),
                             StopFlag(), 2, extra_args={"use_cpu": True, "dataloader_num_workers": 0,
                                                        "save_strategy": "no", "eval_strategy": "no"},
                             contract=False)
    probe = _Probe()
    trainer.add_callback(probe.callback)
    trainer.train()
    assert probe.seen and set(probe.seen) == {ckpt}


def test_train_sampler_groups_by_duration(real_processor, run_dir):
    from transformers.trainer_pt_utils import LengthGroupedSampler

    from app.core.backend_kit import StopFlag

    durs = [0.5, 9.0, 1.0, 8.5, 0.7, 9.5, 1.2, 8.0]
    rd = run_dir("asr", MODEL, {"max_steps": 2, "batch_size": 2})
    ds = q.RowsDataset([{"audio_path": "x", "text": "t", "duration": d, "sample_id": str(i)}
                        for i, d in enumerate(durs)])
    assert ds.lengths() == [int(d * 100) for d in durs]
    trainer = q.make_trainer(read_config(rd), rd, None, tiny_model(), real_processor, ds, ds, StopFlag(), 2,
                             extra_args={"use_cpu": True}, contract=False)
    sampler = trainer._get_train_sampler()
    assert isinstance(sampler, LengthGroupedSampler)
    assert sampler.lengths == ds.lengths()


def test_split_long_rows():
    kept, n = q.split_long([{"duration_s": 30.0}, {"duration_s": 30.01}, {"duration_s": 2}], probe=None)
    assert n == 1 and [r["duration_s"] for r in kept] == [30.0, 2]


def test_eval_skips_and_counts_long_clips(run_dir, tmp_path, monkeypatch):
    """Eval rows over MAX_SECONDS are skipped like in training and counted in the results."""
    from app.core.contract import read_eval_results

    rd = run_dir("asr", MODEL, {"eval": {"baseline": False}})
    final = rd.checkpoints_dir / "final"
    final.mkdir(parents=True)
    (final / ".complete").write_text("")
    (tmp_path / "s").mkdir()
    (tmp_path / "l").mkdir()
    short = _wav_rows(tmp_path / "s", [1.0])[0]["audio_path"]
    long_ = _wav_rows(tmp_path / "l", [31.0])[0]["audio_path"]
    rows = [{"sample_id": "a", "audio_path": short, "text": "x", "duration_s": 5.0},
            {"sample_id": "b", "audio_path": short, "text": "x", "duration_s": 31.0},
            {"sample_id": "c", "audio_path": short, "text": "x", "duration_s": None},  # probed: 1 s
            {"sample_id": "d", "audio_path": long_, "text": "x", "duration_s": None}]  # probed: 31 s
    seen = []
    monkeypatch.setattr(q, "eval_rows", lambda c: rows)
    monkeypatch.setattr(q, "_run_eval", lambda src, rd, rep, rws, prefix, ms: seen.append(rws) or {"wer": 0.0})
    assert q.eval(rd, read_config(rd), None) == 0
    assert [r["sample_id"] for r in seen[0]] == ["a", "c"]
    assert read_eval_results(rd).metrics["skipped_long"] == 2


# ---------------------------------------------------------------- heavy (GPU VM)

@pytest.mark.heavy
def test_qwen_asr_20_steps(run_dir, monkeypatch):
    """GPU VM only: prepare, 20 steps of full fine-tuning with gradient checkpointing on, eval with baseline,
    safetensors export that reloads and transcribes; batched == single-clip transcripts. Prints peak memory."""
    import torch
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    from app.core.backend_kit import Reporter
    from app.core.contract import read_eval_results, read_prepared_refs

    probe = _Probe()
    make = q.make_trainer

    def make_with_probe(*a, **kw):
        trainer = make(*a, **kw)
        trainer.add_callback(probe.callback)
        return trainer

    monkeypatch.setattr(q, "make_trainer", make_with_probe)
    rd = run_dir("asr", MODEL, {"max_steps": 20, "batch_size": 2, "grad_accum": 1, "learning_rate": 1e-5,
                                "eval": {"max_samples": 4}}, texts=BN, seconds=1.5,
                 method={"type": "full"}, backend="qwen-asr", export=["safetensors"])
    cfg = read_config(rd)
    rep = Reporter(rd)
    assert q.prepare(rd, cfg, rep) == 0
    assert read_prepared_refs(rd)[0]["role"] == "train"
    torch.cuda.reset_peak_memory_stats()
    assert q.train(rd, cfg, rep) == 0
    print(f"train peak torch.cuda.max_memory_allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    assert probe.seen and set(probe.seen) == {True}  # gradient checkpointing really on
    final = rd.checkpoints_dir / "final"
    assert (final / ".complete").exists()
    assert json.loads((final / "init_meta.json").read_text())["format"] == "hf-full"
    assert q.eval(rd, cfg, rep) == 0
    res = read_eval_results(rd).metrics
    assert {"wer", "cer", "rtf", "base/wer", "base/cer", "skipped_long"} <= set(res)
    assert q.export(rd, cfg, rep) == 0
    out = rd.export_dir / "safetensors"
    assert list(out.glob("*.safetensors")) and rd.export_manifest_path.exists()

    model = Qwen3ASRForConditionalGeneration.from_pretrained(str(out), dtype=torch.bfloat16).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(str(out))
    clip = next(iter(read_prepared_refs(rd)))["path"]
    wavs = sorted((Path(clip) / "audio" / "train").glob("*.wav"))
    tr = q.make_transcriber(model, processor)
    texts = tr([str(w) for w in wavs])
    assert len(texts) == len(wavs) and all(isinstance(t, str) for t in texts)
    assert texts == [tr([str(w)])[0] for w in wavs]  # batched (left padded) == one clip at a time
