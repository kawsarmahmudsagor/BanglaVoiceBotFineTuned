"""LLM family phases for the unsloth backend: prepare and SFT training (LoRA / QLoRA / full).

torch, transformers, unsloth, trl and datasets are imported inside functions only.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from app.core.backend_kit import (
    NanGuard, StopFlag, init_source, latest_checkpoint, mark_complete, write_init_meta,
)
from app.core.contract import RunConfig, RunDir, read_prepared_refs

from .prepare.chat import prepare_llm as prepare  # noqa: F401

BACKEND = "unsloth"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
InitPlan = Literal["base", "attach_adapter", "merge_adapter", "full_weights", "full_then_adapter"]


def plan_init(meta: dict | None, method: str) -> InitPlan:
    """How to build the starting model from an init checkpoint's meta (spec 14.2)."""
    if meta is None:
        return "base"
    fmt = meta.get("format")
    if fmt == "hf-peft":
        return "merge_adapter" if method == "full" else "attach_adapter"
    if fmt == "hf-full":
        return "full_weights" if method == "full" else "full_then_adapter"
    raise ValueError(f"unsupported init checkpoint format '{fmt}' for backend {BACKEND}")


def _method(cfg: RunConfig) -> str:
    return str(cfg.method.get("type", "lora"))


def _train_ref(rd: RunDir) -> dict:
    for r in read_prepared_refs(rd):
        if r.get("role") == "train":
            return r
    raise RuntimeError("no prepared train dataset (prepared_refs.json has no train entry)")


def total_steps(cfg: RunConfig, rd: RunDir) -> int:
    hp = cfg.hparams
    if hp.get("max_steps"):
        return int(hp["max_steps"])
    n = int(_train_ref(rd).get("counts", {}).get("train", 0))
    per_step = int(hp.get("batch_size", 2)) * int(hp.get("grad_accum", 4))
    return max(1, math.ceil(n / per_step) * math.ceil(float(hp.get("epochs", 1))))


def _trainer_args(cfg: RunConfig, rd: RunDir) -> dict:
    """TrainingArguments kwargs as a plain dict (max_steps overrides num_train_epochs)."""
    hp = cfg.hparams
    method = _method(cfg)
    save_steps = max(50, total_steps(cfg, rd) // 10)
    args: dict = {
        "output_dir": str(rd.checkpoints_dir),
        "per_device_train_batch_size": int(hp.get("batch_size", 2)),
        "per_device_eval_batch_size": int(hp.get("batch_size", 2)),
        "gradient_accumulation_steps": int(hp.get("grad_accum", 4)),
        "learning_rate": float(hp.get("learning_rate", 2e-4)),
        "warmup_steps": int(hp.get("warmup_steps", 5)),
        "save_steps": save_steps,
        "save_total_limit": 3,
        "logging_steps": 10,
        "eval_strategy": "steps",
        "eval_steps": save_steps,
        "bf16": True,
        "optim": "adamw_8bit" if method == "full" else "adamw_torch",
        "report_to": [],
        "seed": cfg.seed,
        "remove_unused_columns": False,
    }
    if hp.get("max_steps"):
        args["max_steps"] = int(hp["max_steps"])
    else:
        args["num_train_epochs"] = float(hp.get("epochs", 1))
    return args


def merged_init_dir(rd: RunDir) -> Path:
    """Scratch dir for a merged hf-peft init (never rd.init_dir, which is read-only)."""
    return rd.path / "work" / "merged_init"


def _require_trainable(model) -> None:
    if not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("model has no trainable parameters; check the init checkpoint and method")


def write_ckpt_meta(ckpt_dir: Path, cfg: RunConfig, step: int) -> None:
    method = _method(cfg)
    write_init_meta(ckpt_dir, backend=BACKEND, base_model=cfg.base_model, method=method,
                    format="hf-full" if method == "full" else "hf-peft", step=step)


def _load_model(cfg: RunConfig, rd: RunDir, max_seq_len: int):
    from unsloth import FastLanguageModel

    method = _method(cfg)
    choice = init_source(cfg, rd)
    # A resumed run restores weights from its own checkpoint; build the same starting shape as the original.
    if choice is not None and choice.kind == "resume":
        choice = init_source(cfg.model_copy(update={"resume": False}), rd)
    plan = plan_init(choice.meta, method) if choice is not None else "base"
    src = str(choice.path) if choice is not None else cfg.base_model  # init_dir is read-only: load only

    def load(name, *, four_bit, full):
        return FastLanguageModel.from_pretrained(
            name, max_seq_length=max_seq_len, load_in_4bit=four_bit, full_finetuning=full)

    def add_adapter(m):
        return FastLanguageModel.get_peft_model(
            m, r=int(cfg.hparams.get("lora_r", 16)), lora_alpha=int(cfg.hparams.get("lora_alpha", 16)),
            target_modules=TARGET_MODULES, use_gradient_checkpointing="unsloth", random_state=cfg.seed)

    four_bit = method == "qlora"
    if plan == "base":
        model, tok = load(cfg.base_model, four_bit=four_bit, full=method == "full")
        if method != "full":
            model = add_adapter(model)
    elif plan == "attach_adapter":
        model, tok = load(src, four_bit=four_bit, full=False)
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)
        _require_trainable(model)
    elif plan == "merge_adapter":
        model, tok = load(src, four_bit=False, full=False)
        model = model.merge_and_unload()
        scratch = merged_init_dir(rd)
        scratch.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(scratch))
        tok.save_pretrained(str(scratch))
        del model
        model, tok = load(str(scratch), four_bit=False, full=True)
    elif plan == "full_weights":
        model, tok = load(src, four_bit=False, full=True)
    else:  # full_then_adapter
        model, tok = load(src, four_bit=four_bit, full=False)
        model = add_adapter(model)
    return model, tok


def train_llm(rd: RunDir, cfg: RunConfig, rep) -> int:
    import unsloth  # noqa: F401  (must precede transformers/trl/datasets imports)
    from datasets import load_dataset
    from transformers import DataCollatorForSeq2Seq
    from trl import SFTConfig, SFTTrainer

    from app.core.backend_kit.hf import ContractCallback

    stop = StopFlag().install()
    max_seq_len = int(cfg.hparams.get("max_seq_len", 2048))
    model, tok = _load_model(cfg, rd, max_seq_len)

    _require_trainable(model)
    prepared = Path(_train_ref(rd)["path"])
    data = load_dataset("json", data_files={"train": str(prepared / "train.jsonl"),
                                            "val": str(prepared / "val.jsonl")})
    total = total_steps(cfg, rd)

    class _Callback(ContractCallback):
        def on_save(self, args, state, control, **kwargs):
            # init_meta must exist before `.complete` appears.
            write_ckpt_meta(Path(args.output_dir) / f"checkpoint-{state.global_step}", cfg, state.global_step)
            super().on_save(args, state, control, **kwargs)

    sft_args = SFTConfig(**_trainer_args(cfg, rd), max_length=max_seq_len,
                         dataset_kwargs={"skip_prepare_dataset": True})
    trainer = SFTTrainer(
        model=model, args=sft_args, train_dataset=data["train"], eval_dataset=data["val"],
        processing_class=tok, data_collator=DataCollatorForSeq2Seq(tok, label_pad_token_id=-100),
        callbacks=[_Callback(rep, stop, NanGuard(), total)],
    )
    last = latest_checkpoint(rd) if cfg.resume else None
    trainer.train(resume_from_checkpoint=str(last) if last else None)
    if stop.requested:
        return 143  # the callback already saved a complete checkpoint; resume continues from it

    final = rd.checkpoints_dir / "final"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    write_ckpt_meta(final, cfg, trainer.state.global_step)
    mark_complete(final)
    return 0
