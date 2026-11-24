"""LLM family phases for the unsloth backend: prepare and SFT training (LoRA / QLoRA / full).

torch, transformers, unsloth, trl and datasets are imported inside functions only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from app.core.backend_kit import (
    NanGuard, StopFlag, eval_opts, init_source, latest_checkpoint, mark_complete, write_init_meta,
)
from app.core.backend_kit.runutil import eval_ds as _eval_ds  # noqa: F401
from app.core.backend_kit.runutil import eval_loss_files, eval_rows  # noqa: F401
from app.core.backend_kit.runutil import free_gpu as _free_gpu  # noqa: F401
from app.core.backend_kit.runutil import stopped_early, total_steps, train_ref as _train_ref, trainer_args  # noqa: F401
from app.core.contract import RunConfig, RunDir
from app.core.evaluation import baseline_metrics, merge_results, run_llm_eval

from .export import export  # noqa: F401  (phase handler looked up by phases._dispatch)
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


def _trainer_args(cfg: RunConfig, rd: RunDir) -> dict:
    return trainer_args(cfg, rd, "adamw_8bit" if _method(cfg) == "full" else "adamw_torch")


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
    if stopped_early(stop, trainer):
        return 143  # the callback already saved a complete checkpoint; resume continues from it

    final = rd.checkpoints_dir / "final"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    write_ckpt_meta(final, cfg, trainer.state.global_step)
    mark_complete(final)
    return 0


def _eval_loss(model, tok, rd: RunDir, cfg: RunConfig) -> float | None:
    from datasets import load_dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    files = [str(f) for f in eval_loss_files(rd, cfg) if f.exists() and f.stat().st_size > 0]
    if not files:
        return None
    data = load_dataset("json", data_files={"eval": files})["eval"]
    args = TrainingArguments(
        output_dir=str(rd.path / "work" / "eval_tmp"), per_device_eval_batch_size=int(cfg.hparams.get("batch_size", 2)),
        bf16=True, report_to=[], remove_unused_columns=False, seed=cfg.seed)
    trainer = Trainer(model=model, args=args, eval_dataset=data, processing_class=tok,
                      data_collator=DataCollatorForSeq2Seq(tok, label_pad_token_id=-100))
    return float(trainer.evaluate()["eval_loss"])


def _make_generate(model, tok):
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)

    def generate(prompt: list[dict]) -> str:
        enc = tok.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True,
                                      return_tensors="pt", return_dict=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=512, do_sample=False)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate


def _eval_one(model, tok, rd, cfg, rep, rows, *, prefix: str, max_samples: int) -> dict[str, float]:
    loss = _eval_loss(model, tok, rd, cfg)
    return run_llm_eval(rd, rep, rows, _make_generate(model, tok), loss=loss, prefix=prefix, max_samples=max_samples)


def _baseline_eval_key(cfg: RunConfig, max_samples: int, max_seq_len: int) -> dict:
    key: dict = {"max_samples": max_samples, "max_seq_len": max_seq_len,
                 "load_in_4bit": _method(cfg) == "qlora"}
    if cfg.init_from is not None:
        key["init"] = [cfg.init_from.source_run, cfg.init_from.checkpoint]
    return key


def eval_llm(rd: RunDir, cfg: RunConfig, rep) -> int:
    final = rd.checkpoints_dir / "final"
    if not (final / ".complete").exists():
        raise RuntimeError(f"no completed final checkpoint at {final}")
    import unsloth  # noqa: F401  (must precede transformers/datasets imports)
    from unsloth import FastLanguageModel

    do_baseline, user_max = eval_opts(cfg)
    max_samples = max(1, min(200, user_max)) if user_max is not None else 200
    max_seq_len = int(cfg.hparams.get("max_seq_len", 2048))
    rows = eval_rows(cfg)

    model, tok = FastLanguageModel.from_pretrained(
        str(final), max_seq_length=max_seq_len, load_in_4bit=_method(cfg) == "qlora")
    tuned = _eval_one(model, tok, rd, cfg, rep, rows, prefix="", max_samples=max_samples)
    del model, tok
    _free_gpu()

    base: dict[str, float] = {}
    if do_baseline:
        ds, _ = _eval_ds(cfg)
        init = cfg.init_from
        eval_key = _baseline_eval_key(cfg, max_samples, max_seq_len)

        def compute() -> dict[str, float]:
            if init is not None:
                m, t = _load_model(cfg.model_copy(update={"resume": False}), rd, max_seq_len)
            else:
                m, t = FastLanguageModel.from_pretrained(
                    cfg.base_model, max_seq_length=max_seq_len, load_in_4bit=_method(cfg) == "qlora")
            try:
                res = _eval_one(m, t, rd, cfg, rep, rows, prefix="base/", max_samples=max_samples)
                # baseline_metrics adds the `base/` prefix itself; prefix here only skips sample writing.
                return {k.removeprefix("base/"): v for k, v in res.items()}
            finally:
                del m, t
                _free_gpu()

        base = baseline_metrics(cfg, ds, eval_key, compute)
    merge_results(rd, tuned, base)
    return 0


train = train_llm
eval = eval_llm  # noqa: A001  (phase handler name looked up by phases._dispatch)
