"""Whisper full fine-tune for the hf backend (plain transformers): train, eval, export.

torch, transformers and datasets are imported inside functions only. Prepared data, collator, transcriber,
eval loss, baseline key and the ct2 command are shared with the unsloth LoRA path (app.core.backend_kit).

Transformers compatibility: this venv pins 4.46.1 (parler-tts) while unsloth runs 5.x. The only differing
API used here is the Trainer processor kwarg, resolved by signature in backend_kit.whisper.processor_kwarg.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

from app.core.backend_kit import (
    NanGuard, StopFlag, eval_opts, export_all, init_source, latest_checkpoint, mark_complete, write_init_meta,
)
from app.core.backend_kit.runutil import (
    dir_artifact, eval_ds, eval_loss_files, eval_rows, free_gpu, total_steps, trainer_args,
)
from app.core.backend_kit.whisper import (
    WhisperCollator, asr_baseline_key, check_prepared_refs, dataloader_workers, load_processor, load_train_val,
    make_transcriber, prepare_whisper, processor_kwarg, run_ct2, whisper_eval_loss,
)
from app.core.contract import RunConfig, RunDir
from app.core.evaluation import baseline_metrics, merge_results, run_asr_eval

BACKEND = "hf"
InitPlan = Literal["base", "full_weights"]

prepare = prepare_whisper  # phase handler looked up by phases._dispatch


def plan_whisper_init(meta: dict | None) -> InitPlan:
    """Starting model from an init checkpoint's meta. A full fine-tune can only continue from full weights."""
    if meta is None:
        return "base"
    fmt = meta.get("format")
    if fmt == "hf-full":
        return "full_weights"
    raise ValueError(f"unsupported init checkpoint format '{fmt}' for a full whisper fine-tune on backend {BACKEND}: "
                     "only hf-full checkpoints can be used (merge adapters into full weights first)")


def write_ckpt_meta(ckpt_dir: Path, cfg: RunConfig, step: int) -> None:
    write_init_meta(ckpt_dir, backend=BACKEND, base_model=cfg.base_model, method="full", format="hf-full", step=step)


def whisper_trainer_args(cfg: RunConfig, rd: RunDir) -> dict:
    args = trainer_args(cfg, rd, "adamw_torch")
    args["label_names"] = ["labels"]
    args["predict_with_generate"] = False
    args["dataloader_num_workers"] = dataloader_workers()
    args["gradient_checkpointing"] = True
    args["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return args


def _processor(cfg: RunConfig):
    return load_processor(cfg.base_model)


def _load(src: str, *, bf16: bool = False):
    import torch
    from transformers import WhisperForConditionalGeneration

    kw = {"torch_dtype": torch.bfloat16} if bf16 else {}
    model = WhisperForConditionalGeneration.from_pretrained(src, **kw)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    return model


def _start_path(cfg: RunConfig, rd: RunDir) -> str:
    """Weights to start from: the warm-start init dir (read-only, load only) or the base model.
    A resumed run starts from the same source; the Trainer then restores its own checkpoint over it."""
    choice = init_source(cfg.model_copy(update={"resume": False}), rd)
    if choice is None:
        return cfg.base_model
    plan_whisper_init(choice.meta)
    return str(choice.path)


def train_whisper(rd: RunDir, cfg: RunConfig, rep) -> int:
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    from app.core.backend_kit.hf import MetaContractCallback

    stop = StopFlag().install()
    model = _load(_start_path(cfg, rd))
    model.config.use_cache = False  # incompatible with gradient checkpointing
    processor = _processor(cfg)
    train_ds, val_ds, _ = load_train_val(rd)
    total = total_steps(cfg, rd)

    trainer = Seq2SeqTrainer(
        model=model, args=Seq2SeqTrainingArguments(**whisper_trainer_args(cfg, rd)),
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=WhisperCollator(processor, int(model.config.decoder_start_token_id)),
        callbacks=[MetaContractCallback(rep, stop, NanGuard(), total, lambda d, step: write_ckpt_meta(d, cfg, step))],
        **processor_kwarg(Seq2SeqTrainer, processor.feature_extractor),
    )
    last = latest_checkpoint(rd) if cfg.resume else None
    trainer.train(resume_from_checkpoint=str(last) if last else None)
    if stop.requested:
        return 143  # the callback already saved a complete checkpoint; resume continues from it

    final = rd.checkpoints_dir / "final"
    trainer.save_model(str(final))
    processor.save_pretrained(str(final))
    write_ckpt_meta(final, cfg, trainer.state.global_step)  # meta before .complete
    mark_complete(final)
    return 0


def _device(model):
    return getattr(model, "device", "cuda")


def _eval_loss(model, cfg: RunConfig, rd: RunDir) -> float | None:
    dirs = [f.with_suffix("") for f in eval_loss_files(rd, cfg)]
    return whisper_eval_loss(model, _processor(cfg), dirs, batch_size=int(cfg.hparams.get("batch_size", 2)),
                             seed=cfg.seed, work_dir=rd.path / "work" / "eval_tmp")


def _eval_model(model, cfg: RunConfig, rd: RunDir, rep, rows, prefix: str, max_samples) -> dict[str, float]:
    model.to("cuda").eval()
    loss = _eval_loss(model, cfg, rd)
    res = run_asr_eval(rd, rep, rows, make_transcriber(model, _processor(cfg), _device(model)),
                       prefix=prefix, max_samples=max_samples)
    if loss is not None:
        res[f"{prefix}eval_loss"] = loss
    return res


def eval_whisper(rd: RunDir, cfg: RunConfig, rep) -> int:
    final = rd.checkpoints_dir / "final"
    if not (final / ".complete").exists():
        raise RuntimeError(f"no completed final checkpoint at {final}")
    do_baseline, max_samples = eval_opts(cfg)
    rows = eval_rows(cfg)
    check_prepared_refs(rd)

    model = _load(str(final), bf16=True)
    tuned = _eval_model(model, cfg, rd, rep, rows, "", max_samples)
    del model
    free_gpu()

    base: dict[str, float] = {}
    if do_baseline:
        ds, _ = eval_ds(cfg)

        def compute() -> dict[str, float]:
            m = _load(_start_path(cfg, rd), bf16=True)  # init checkpoint when set, else the base model
            try:
                res = _eval_model(m, cfg, rd, rep, rows, "base/", max_samples)
                return {k.removeprefix("base/"): v for k, v in res.items()}  # baseline_metrics re-adds `base/`
            finally:
                del m
                free_gpu()

        base = baseline_metrics(cfg, ds, asr_baseline_key(cfg, max_samples), compute)
    merge_results(rd, tuned, base)
    return 0


def whisper_exporters(rd: RunDir, cfg: RunConfig) -> dict:
    final = rd.checkpoints_dir / "final"
    state: dict = {}

    def save_into(out: Path) -> None:
        if not (final / ".complete").exists():
            raise RuntimeError(f"no completed final checkpoint at {final}")
        model = _load(str(final))
        model.save_pretrained(str(out), safe_serialization=True)
        _processor(cfg).save_pretrained(str(out))  # ct2 --copy_files needs tokenizer.json + preprocessor_config.json

    def safetensors(out: Path):
        save_into(out)
        state["saved"] = out
        return [dir_artifact(rd, "safetensors", out, "transformers")]

    def ct2(out: Path):
        src, tmp = state.get("saved"), None
        if src is None or not (Path(src) / "config.json").exists():
            tmp = rd.path / "work" / "ct2_src"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            save_into(tmp)
            src = tmp
        try:
            run_ct2(Path(src), out)
        finally:
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)
        return [dir_artifact(rd, "ct2", out, "faster-whisper")]

    return {"safetensors": safetensors, "ct2": ct2}


def export_whisper(rd: RunDir, cfg: RunConfig, rep) -> int:
    return export_all(rd, cfg, rep, whisper_exporters(rd, cfg))


train = train_whisper
eval = eval_whisper  # noqa: A001  (phase handler name looked up by phases._dispatch)
export = export_whisper
