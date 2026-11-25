"""Qwen3-ASR full fine-tune (spec 17): prepare, train, eval, export phase handlers.

torch and transformers are imported inside functions only. Run plumbing (ContractCallback, init meta, the eval
harness, baseline cache, export) is shared with the other backends through app.core.backend_kit.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from app.core.backend_kit import (
    NanGuard, StopFlag, eval_opts, export_all, init_source, latest_checkpoint, mark_complete, max_skip_share,
    write_init_meta,
)
from app.core.backend_kit.runutil import (
    cap_val, dir_artifact, eval_ds, eval_rows, free_gpu, stopped_early, total_steps, train_ref, trainer_args,
)
from app.core.contract import RunConfig, RunDir, write_prepared_refs
from app.core.evaluation import baseline_metrics, merge_results, run_asr_eval
from app.core.prepare import ensure_prepared

from .prepare.qwen_asr import (
    MAX_LABEL_TOKENS, build_qwen_asr, check_prepared, qwen_asr_spec, read_rows, tokenizer_hash,
)
from .prompt import (
    SAMPLE_RATE, QwenAsrCollator, generation_inputs, max_new_tokens, prompt_text, split_transcript,
)

BACKEND = "qwen-asr"
InitPlan = Literal["base", "full_weights"]

# spec 17.5 defaults; recipe hparams override them
DEFAULTS: dict = {
    "learning_rate": 2e-5,
    "batch_size": 8,
    "grad_accum": 1,
    "warmup_ratio": 0.02,
    "epochs": 3,
    "max_label_tokens": MAX_LABEL_TOKENS,
    "gradient_checkpointing": True,
    "freeze_audio_encoder": False,
}
EVAL_BATCH = 16


def with_defaults(cfg: RunConfig) -> RunConfig:
    return cfg.model_copy(update={"hparams": {**DEFAULTS, **cfg.hparams}})


def plan_init(meta: dict | None) -> InitPlan:
    """Starting model from an init checkpoint's meta. A full fine-tune continues from full weights only."""
    if meta is None:
        return "base"
    fmt = meta.get("format")
    if fmt == "hf-full":
        return "full_weights"
    raise ValueError(f"unsupported init checkpoint format '{fmt}' for a full Qwen3-ASR fine-tune on backend "
                     f"{BACKEND}: only hf-full checkpoints can be used")


def write_ckpt_meta(ckpt_dir: Path, cfg: RunConfig, step: int) -> None:
    write_init_meta(ckpt_dir, backend=BACKEND, base_model=cfg.base_model, method="full", format="hf-full", step=step)


def start_path(cfg: RunConfig, rd: RunDir) -> str:
    """Weights to start from: the warm-start init dir (read-only, load only) or the base model.
    A resumed run starts from the same source; the Trainer then restores its own checkpoint over it."""
    choice = init_source(cfg.model_copy(update={"resume": False}), rd)
    if choice is None:
        return cfg.base_model
    plan_init(choice.meta)
    return str(choice.path)


def load_processor(src: str):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(src)


def load_model(src: str, *, bf16: bool):
    import torch
    from transformers import Qwen3ASRForConditionalGeneration

    dtype = torch.bfloat16 if bf16 else torch.float32  # training: fp32 master weights, bf16 autocast
    return Qwen3ASRForConditionalGeneration.from_pretrained(src, dtype=dtype, attn_implementation="sdpa")


def freeze_audio_encoder(model) -> int:
    """Freeze the audio tower; the LLM decoder and the audio projector keep training. Returns frozen params."""
    tower = model.model.audio_tower
    tower.requires_grad_(False)
    return sum(p.numel() for p in tower.parameters())


# ---------------------------------------------------------------- prepare

def prepare(rd: RunDir, cfg: RunConfig, rep) -> int:
    processor = load_processor(cfg.base_model)
    tok = processor.tokenizer
    max_tokens = int(with_defaults(cfg).hparams["max_label_tokens"])
    spec = qwen_asr_spec(cfg, tokenizer_hash(tok), max_tokens)
    build = build_qwen_asr(lambda t: tok(t, add_special_tokens=False).input_ids, rep, max_label_tokens=max_tokens)
    # Eval transcribes the source rows (any format) directly, so only the train dataset is converted.
    pi = ensure_prepared(cfg, cfg.datasets["train"], spec, build, splits=["train", "val"],
                         max_skip_share=max_skip_share(cfg))
    write_prepared_refs(rd, [{"role": "train", **pi.model_dump()}])
    return 0


# ---------------------------------------------------------------- data

class RowsDataset:
    """Prepared rows in memory; audio is decoded by the collator."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]

    def select(self, indices) -> "RowsDataset":
        return RowsDataset([self.rows[i] for i in indices])


def load_train_val(rd: RunDir) -> tuple[RowsDataset, RowsDataset]:
    prepared = Path(train_ref(rd)["path"])
    check_prepared(prepared, "train")
    return RowsDataset(read_rows(prepared, "train")), RowsDataset(read_rows(prepared, "val"))


# ---------------------------------------------------------------- train

def qwen_trainer_args(cfg: RunConfig, rd: RunDir) -> dict:
    from app.core.backend_kit.whisper import dataloader_workers

    cfg = with_defaults(cfg)
    hp = cfg.hparams
    args = trainer_args(cfg, rd, "adamw_torch")
    # transformers 5: a float in [0, 1) is a ratio of the total steps (warmup_ratio was folded into warmup_steps)
    args["warmup_steps"] = float(hp["warmup_ratio"])
    if hp.get("save_steps"):
        args["save_steps"] = args["eval_steps"] = int(hp["save_steps"])
    args["label_names"] = ["labels"]
    args["prediction_loss_only"] = True
    args["dataloader_num_workers"] = dataloader_workers()
    # transformers 5.24 renamed gradient_checkpointing(_kwargs) to activation_checkpointing(_kwargs)
    args["activation_checkpointing"] = bool(hp["gradient_checkpointing"])
    args["activation_checkpointing_kwargs"] = {"use_reentrant": False}
    return args


def train(rd: RunDir, cfg: RunConfig, rep) -> int:
    from transformers import Trainer, TrainingArguments

    from app.core.backend_kit.hf import MetaContractCallback

    stop = StopFlag().install()
    hp = with_defaults(cfg).hparams
    src = start_path(cfg, rd)
    model = load_model(src, bf16=False)
    if hp["gradient_checkpointing"]:
        model.config.use_cache = False
        if getattr(model.config, "text_config", None) is not None:
            model.config.text_config.use_cache = False
    if hp["freeze_audio_encoder"]:
        freeze_audio_encoder(model)
    processor = load_processor(src)
    train_ds, val_ds = load_train_val(rd)
    val_ds = cap_val(val_ds, cfg)  # eval every save_steps: a seeded subset, not the whole val split
    total = total_steps(with_defaults(cfg), rd)

    trainer = Trainer(
        model=model, args=TrainingArguments(**qwen_trainer_args(cfg, rd)),
        train_dataset=train_ds, eval_dataset=val_ds, data_collator=QwenAsrCollator(processor),
        callbacks=[MetaContractCallback(rep, stop, NanGuard(), total, lambda d, step: write_ckpt_meta(d, cfg, step))],
        processing_class=processor,
    )
    last = latest_checkpoint(rd) if cfg.resume else None
    trainer.train(resume_from_checkpoint=str(last) if last else None)
    if stopped_early(stop, trainer):
        return 143  # the callback already saved a complete checkpoint; resume continues from it

    final = rd.checkpoints_dir / "final"
    model.config.use_cache = True  # the saved config must be usable for generation downstream
    if getattr(model.config, "text_config", None) is not None:
        model.config.text_config.use_cache = True
    trainer.save_model(str(final))
    processor.save_pretrained(str(final))
    write_ckpt_meta(final, cfg, trainer.state.global_step)  # meta before .complete
    mark_complete(final)
    return 0


# ---------------------------------------------------------------- eval

def _loader(path):
    from app.core.backend_kit.audio import load_audio

    return load_audio(path, SAMPLE_RATE)


def make_transcriber(model, processor, loader: Callable = _loader) -> Callable[[list[str]], list[str]]:
    """Batched greedy transcription with the PREFIX prefilled (continue_final_message) and left padding."""
    prompt = prompt_text(processor)

    def transcribe(paths: list[str]) -> list[str]:
        import torch

        audios = [loader(p) for p in paths]
        inputs = generation_inputs(processor, prompt, audios, model.device, model.dtype)
        longest = max(len(a) for a in audios) / SAMPLE_RATE
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens(longest), do_sample=False)
        texts = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return [split_transcript(t) for t in texts]

    return transcribe


def _require_final(rd: RunDir) -> Path:
    final = rd.checkpoints_dir / "final"
    if not (final / ".complete").exists():
        raise RuntimeError(f"no completed final checkpoint at {final}")
    return final


def _run_eval(src: str, rd: RunDir, rep, rows, prefix: str, max_samples) -> dict[str, float]:
    model = load_model(src, bf16=True).to("cuda").eval()
    try:
        processor = load_processor(src)
        return run_asr_eval(rd, rep, rows, make_transcriber(model, processor), prefix=prefix,
                            batch_size=EVAL_BATCH, max_samples=max_samples)
    finally:
        del model
        free_gpu()


def eval(rd: RunDir, cfg: RunConfig, rep) -> int:  # noqa: A001 - phase handler name looked up by phases._dispatch
    from app.core.backend_kit.whisper import asr_baseline_key

    final = _require_final(rd)
    do_baseline, max_samples = eval_opts(cfg)
    rows = eval_rows(cfg)
    tuned = _run_eval(str(final), rd, rep, rows, "", max_samples)

    base: dict[str, float] = {}
    if do_baseline:
        ds, _ = eval_ds(cfg)

        def compute() -> dict[str, float]:
            # the untouched starting model: the init checkpoint when set, else the base model
            res = _run_eval(start_path(cfg, rd), rd, rep, rows, "base/", max_samples)
            return {k.removeprefix("base/"): v for k, v in res.items()}  # baseline_metrics re-adds `base/`

        base = baseline_metrics(cfg, ds, asr_baseline_key(cfg, max_samples), compute)
    merge_results(rd, tuned, base)
    return 0


# ---------------------------------------------------------------- export

def exporters(rd: RunDir, cfg: RunConfig) -> dict:
    def safetensors(out: Path):
        final = _require_final(rd)
        model = load_model(str(final), bf16=True)  # bf16, like the base model on the Hub
        model.save_pretrained(str(out), safe_serialization=True)
        load_processor(str(final)).save_pretrained(str(out))
        del model
        return [dir_artifact(rd, "safetensors", out, "transformers")]

    return {"safetensors": safetensors}


def export(rd: RunDir, cfg: RunConfig, rep) -> int:
    return export_all(rd, cfg, rep, exporters(rd, cfg))
