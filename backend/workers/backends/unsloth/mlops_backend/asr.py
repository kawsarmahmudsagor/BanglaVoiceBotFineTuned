"""Whisper (ASR) family phases for the unsloth backend: LoRA training, eval and export.

torch, transformers, unsloth, trl and datasets are imported inside functions only.
Start-from-checkpoint rules (plan_init, init_meta, read-only init dir) are shared with the LLM path.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from app.core.backend_kit import (
    NanGuard, StopFlag, eval_opts, export_all, init_source, latest_checkpoint, mark_complete,
)
from app.core.backend_kit.whisper import (
    LANGUAGE, TASK, make_transcriber, pad_labels, prepare_whisper,
)
from app.core.contract import RunConfig, RunDir
from app.core.evaluation import baseline_metrics, merge_results, run_asr_eval

from .export import _dir_artifact
from .llm import (
    _eval_ds, _free_gpu, _require_trainable, _train_ref, _trainer_args, eval_rows, plan_init,
    total_steps, write_ckpt_meta,
)

TARGET_MODULES_ASR = ["q_proj", "v_proj"]
prepare = prepare_whisper  # noqa: F841  (phase handler looked up by phases._dispatch)


def asr_trainer_args(cfg: RunConfig, rd: RunDir) -> dict:
    args = _trainer_args(cfg, rd)
    args["label_names"] = ["labels"]
    args["predict_with_generate"] = False
    return args


def _processor(cfg: RunConfig):
    from transformers import WhisperProcessor

    p = WhisperProcessor.from_pretrained(cfg.base_model)
    p.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)
    return p


def _from_pretrained(name: str):
    from transformers import WhisperForConditionalGeneration
    from unsloth import FastModel

    return FastModel.from_pretrained(
        name, auto_model=WhisperForConditionalGeneration, whisper_language="Bengali", whisper_task=TASK,
        load_in_4bit=False, full_finetuning=False)


def _load_asr(cfg: RunConfig, rd: RunDir):
    """Model for training (or for the warm-start baseline) following plan_init; returns a PEFT model."""
    from unsloth import FastModel

    choice = init_source(cfg, rd)
    if choice is not None and choice.kind == "resume":  # same starting shape as the original run
        choice = init_source(cfg.model_copy(update={"resume": False}), rd)
    plan = plan_init(choice.meta, "lora") if choice is not None else "base"
    src = str(choice.path) if choice is not None else cfg.base_model  # init_dir is read-only: load only

    def add_adapter(m):
        return FastModel.get_peft_model(
            m, r=int(cfg.hparams.get("lora_r", 16)), lora_alpha=int(cfg.hparams.get("lora_alpha", 16)),
            target_modules=TARGET_MODULES_ASR, use_gradient_checkpointing="unsloth", random_state=cfg.seed,
            task_type=None)

    if plan == "attach_adapter":
        model, _ = _from_pretrained(src)
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)
        _require_trainable(model)
    else:  # base | full_then_adapter
        model, _ = _from_pretrained(src)
        model = add_adapter(model)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    return model


def _make_collator(processor, decoder_start_id: int):
    import numpy as np
    import torch

    def collate(features: list[dict]) -> dict:
        feats = np.stack([np.asarray(f["input_features"], dtype=np.float32) for f in features])
        labels = pad_labels([list(map(int, f["labels"])) for f in features], decoder_start_id)
        return {"input_features": torch.from_numpy(feats), "labels": torch.tensor(labels, dtype=torch.long)}

    return collate


def train_asr(rd: RunDir, cfg: RunConfig, rep) -> int:
    import unsloth  # noqa: F401  (must precede transformers/trl/datasets imports)
    from datasets import load_from_disk
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    from app.core.backend_kit.hf import ContractCallback

    stop = StopFlag().install()
    model = _load_asr(cfg, rd)
    _require_trainable(model)
    processor = _processor(cfg)
    prepared = Path(_train_ref(rd)["path"])
    train_ds = load_from_disk(str(prepared / "train")).with_format("numpy")
    val_ds = load_from_disk(str(prepared / "val")).with_format("numpy")
    total = total_steps(cfg, rd)

    class _Callback(ContractCallback):
        def on_save(self, args, state, control, **kwargs):
            # init_meta must exist before `.complete` appears.
            write_ckpt_meta(Path(args.output_dir) / f"checkpoint-{state.global_step}", cfg, state.global_step)
            super().on_save(args, state, control, **kwargs)

    trainer = Seq2SeqTrainer(
        model=model, args=Seq2SeqTrainingArguments(**asr_trainer_args(cfg, rd)),
        train_dataset=train_ds, eval_dataset=val_ds, processing_class=processor.feature_extractor,
        data_collator=_make_collator(processor, int(model.config.decoder_start_token_id)),
        callbacks=[_Callback(rep, stop, NanGuard(), total)],
    )
    last = latest_checkpoint(rd) if cfg.resume else None
    trainer.train(resume_from_checkpoint=str(last) if last else None)
    if stop.requested:
        return 143  # the callback already saved a complete checkpoint; resume continues from it

    final = rd.checkpoints_dir / "final"
    trainer.save_model(str(final))
    processor.save_pretrained(str(final))
    write_ckpt_meta(final, cfg, trainer.state.global_step)
    mark_complete(final)
    return 0


def _device(model):
    return getattr(model, "device", "cuda")


def _transcriber(model, cfg: RunConfig):
    try:
        from unsloth import FastModel

        FastModel.for_inference(model)
    except Exception:  # noqa: BLE001 - for_inference is an optimisation only
        model.eval()
    return make_transcriber(model, _processor(cfg), _device(model))


def _asr_baseline_key(cfg: RunConfig, max_samples: int | None) -> dict:
    key: dict = {"max_samples": max_samples}
    if cfg.init_from is not None:
        key["init"] = [cfg.init_from.source_run, cfg.init_from.checkpoint]
    return key


def eval_asr(rd: RunDir, cfg: RunConfig, rep) -> int:
    final = rd.checkpoints_dir / "final"
    if not (final / ".complete").exists():
        raise RuntimeError(f"no completed final checkpoint at {final}")
    import unsloth  # noqa: F401  (must precede transformers/datasets imports)

    do_baseline, max_samples = eval_opts(cfg)
    rows = eval_rows(cfg)

    model, _ = _from_pretrained(str(final))
    tuned = run_asr_eval(rd, rep, rows, _transcriber(model, cfg), max_samples=max_samples)
    del model
    _free_gpu()

    base: dict[str, float] = {}
    if do_baseline:
        ds, _ = _eval_ds(cfg)

        def compute() -> dict[str, float]:
            if cfg.init_from is not None:
                m = _load_asr(cfg.model_copy(update={"resume": False}), rd)
            else:
                m, _ = _from_pretrained(cfg.base_model)
            try:
                res = run_asr_eval(rd, rep, rows, _transcriber(m, cfg), prefix="base/", max_samples=max_samples)
                return {k.removeprefix("base/"): v for k, v in res.items()}  # baseline_metrics re-adds `base/`
            finally:
                del m
                _free_gpu()

        base = baseline_metrics(cfg, ds, _asr_baseline_key(cfg, max_samples), compute)
    merge_results(rd, tuned, base)
    return 0


def ct2_command(merged: Path, out: Path) -> list[str]:
    """ct2-transformers-converter from this interpreter's environment (never a bare PATH lookup)."""
    script = Path(sys.executable).parent / ("ct2-transformers-converter.exe" if sys.platform == "win32"
                                            else "ct2-transformers-converter")
    head = [str(script)] if script.exists() else [sys.executable, "-m", "ctranslate2.converters.transformers"]
    return head + ["--model", str(merged), "--output_dir", str(out), "--force", "--quantization", "float16",
                   "--copy_files", "tokenizer.json", "preprocessor_config.json"]


def asr_exporters(rd: RunDir, cfg: RunConfig) -> dict:
    final = rd.checkpoints_dir / "final"
    state: dict = {}

    def loaded():
        if "m" not in state:
            if not (final / ".complete").exists():
                raise RuntimeError(f"no completed final checkpoint at {final}")
            import unsloth  # noqa: F401  (must precede transformers imports)

            state["m"] = _from_pretrained(str(final))[0]
        return state["m"]

    def merge_into(out: Path) -> None:
        model = loaded()
        processor = _processor(cfg)
        model.save_pretrained_merged(str(out), processor, save_method="merged_16bit")
        if not ((out / "tokenizer.json").exists() and (out / "preprocessor_config.json").exists()):
            processor.save_pretrained(str(out))  # ct2 --copy_files needs both

    def merged(out: Path):
        merge_into(out)
        state["merged_out"] = out
        return [_dir_artifact(rd, "merged-safetensors", out, "transformers")]

    def ct2(out: Path):
        src = state.get("merged_out")
        tmp = None
        if src is None or not (Path(src) / "config.json").exists():
            tmp = rd.path / "work" / "ct2_merge"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            merge_into(tmp)
            src = tmp
        try:
            subprocess.run(ct2_command(Path(src), out), check=True)
        finally:
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)
        return [_dir_artifact(rd, "ct2", out, "faster-whisper")]

    return {"merged-safetensors": merged, "ct2": ct2}


def export(rd: RunDir, cfg: RunConfig, rep) -> int:
    return export_all(rd, cfg, rep, asr_exporters(rd, cfg))


train = train_asr
eval = eval_asr  # noqa: A001  (phase handler name looked up by phases._dispatch)
