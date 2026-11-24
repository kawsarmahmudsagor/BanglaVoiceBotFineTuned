"""Indic Parler-TTS full fine-tune for the hf backend: prepare, train, eval, export.

torch, transformers, parler_tts and datasets are imported inside functions only. Run plumbing is shared with the
whisper/ctc modules via app.core.backend_kit.

Objective (parler-tts training script): `model(input_ids, attention_mask, prompt_input_ids, prompt_attention_mask,
labels=codes)`, labels = DAC codes (rows, frames+1 EOS, codebooks) padded with -100. The model shifts the labels
right (decoder_start_token_id = BOS) and applies the delay pattern itself. The audio encoder (DAC) is always
frozen; the text encoder (flan-t5 description encoder) is frozen too unless hparams.freeze_text_encoder is false.
RISK: parler-tts is not installed on the dev machine; the EOS-append / -100 padding follows the upstream training
script from memory and is confirmed only by the heavy test on the VM.

Warm start: hf-full init -> from_pretrained(<init path>). rd.init_dir is read-only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from app.core.backend_kit import (
    NanGuard, StopFlag, eval_opts, export_all, init_source, latest_checkpoint, mark_complete, max_skip_share,
    write_init_meta,
)
from app.core.backend_kit.runutil import (
    dir_artifact, eval_ds, eval_loss_files, eval_rows, free_gpu, total_steps, trainer_args, train_ref,
)
from app.core.backend_kit.whisper import asr_baseline_key, dataloader_workers, processor_kwarg
from app.core.contract import RunConfig, RunDir, write_prepared_refs
from app.core.evaluation import baseline_metrics, merge_results, run_tts_eval
from app.core.prepare import ensure_prepared

from .models import MODELS
from .prepare.parler import (
    DEFAULT_TEMPLATE, SAMPLE_RATE, build_parler, describe, pad_code_labels, parler_spec,
)

BACKEND = "hf"
InitPlan = Literal["base", "full_weights"]
SYNTH_SEED = 0


def plan_parler_init(meta: dict | None) -> InitPlan:
    if meta is None:
        return "base"
    fmt = meta.get("format")
    if fmt == "hf-full":
        return "full_weights"
    raise ValueError(f"unsupported init checkpoint format '{fmt}' for a Parler-TTS run on backend {BACKEND}: "
                     "only hf-full checkpoints can be used")


def write_ckpt_meta(ckpt_dir: Path, cfg: RunConfig, step: int) -> None:
    write_init_meta(ckpt_dir, backend=BACKEND, base_model=cfg.base_model, method="full", format="hf-full", step=step)


def template_of(cfg: RunConfig) -> str:
    return str(cfg.hparams.get("description_template", DEFAULT_TEMPLATE))


# ---------------------------------------------------------------- models

def load_tokenizers(base_model: str):
    """(prompt tokenizer, description tokenizer): the prompt one lives in the repo, the description one is the
    text encoder's own tokenizer named in the model config."""
    from parler_tts import ParlerTTSConfig
    from transformers import AutoTokenizer

    conf = ParlerTTSConfig.from_pretrained(base_model)
    return (AutoTokenizer.from_pretrained(base_model),
            AutoTokenizer.from_pretrained(conf.text_encoder._name_or_path))


def _feature_extractor(cfg: RunConfig):
    from transformers import AutoFeatureExtractor

    return AutoFeatureExtractor.from_pretrained(cfg.base_model)


def _load(src: str, *, bf16: bool = False):
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration

    kw = {"torch_dtype": torch.bfloat16} if bf16 else {}
    return ParlerTTSForConditionalGeneration.from_pretrained(src, **kw)


def _start_path(cfg: RunConfig, rd: RunDir) -> str:
    """Warm-start init dir (read-only, load only) or the base model; resume starts from the same source."""
    choice = init_source(cfg.model_copy(update={"resume": False}), rd)
    if choice is None:
        return cfg.base_model
    plan_parler_init(choice.meta)
    return str(choice.path)


def _require_final(rd: RunDir) -> Path:
    final = rd.checkpoints_dir / "final"
    if not (final / ".complete").exists():
        raise RuntimeError(f"no completed final checkpoint at {final}")
    return final


# ---------------------------------------------------------------- prepare

def prepare_parler(rd: RunDir, cfg: RunConfig, rep) -> int:
    info = MODELS[cfg.base_model]
    if info.gated:
        from app.core.backend_kit.hf_access import check_hf_access

        check_hf_access(cfg.base_model)
    template = template_of(cfg)
    spec = parler_spec(cfg, template)
    share = max_skip_share(cfg)

    def build(rows, out):  # models load only on a cache miss
        model = _load(cfg.base_model)
        enc = model.audio_encoder
        del model
        ptok, dtok = load_tokenizers(cfg.base_model)
        try:
            return build_parler(_feature_extractor(cfg), enc, ptok, dtok, template, rep)(rows, out)
        finally:
            del enc
            free_gpu()

    refs = []
    for role, key, splits in (("train", "train", ["train", "val"]), ("eval", "eval", None)):
        ds = cfg.datasets.get(key)
        if ds is None:
            continue
        pi = ensure_prepared(cfg, ds, spec, build, splits=splits, max_skip_share=share)
        refs.append({"role": role, **pi.model_dump()})
    write_prepared_refs(rd, refs)
    return 0


prepare = prepare_parler  # phase handler looked up by phases._dispatch


# ---------------------------------------------------------------- data

class ParlerCollator:
    """Pads description/prompt ids with their tokenizers and the DAC codes (+EOS frame) with -100."""

    def __init__(self, prompt_tokenizer, description_tokenizer, eos_id: int):
        self.ptok = prompt_tokenizer
        self.dtok = description_tokenizer
        self.eos_id = eos_id

    def __call__(self, features: list[dict]) -> dict:
        import torch

        d = self.dtok.pad([{"input_ids": f["input_ids"]} for f in features], return_tensors="pt")
        p = self.ptok.pad([{"input_ids": f["prompt_input_ids"]} for f in features], return_tensors="pt")
        labels = pad_code_labels([(list(f["codes"]), int(f["num_codebooks"])) for f in features], self.eos_id)
        return {"input_ids": d["input_ids"], "attention_mask": d["attention_mask"],
                "prompt_input_ids": p["input_ids"], "prompt_attention_mask": p["attention_mask"],
                "labels": torch.tensor(labels, dtype=torch.long)}


def _split_dirs(rd: RunDir, splits: list[str]) -> list[Path]:
    prepared = Path(train_ref(rd)["path"])
    return [prepared / s for s in splits]


def _load_concat(dirs: list[Path]):
    from datasets import concatenate_datasets, load_from_disk

    parts = [load_from_disk(str(d)) for d in dirs]
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def trainer_cls():
    from transformers import Trainer

    class ParlerTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            out = model(**inputs)  # input_ids, attention_mask, prompt_input_ids, prompt_attention_mask, labels
            return (out.loss, out) if return_outputs else out.loss

    return ParlerTrainer


def freeze_encoders(model, freeze_text: bool) -> None:
    model.audio_encoder.requires_grad_(False)
    if freeze_text:
        model.text_encoder.requires_grad_(False)


def parler_trainer_args(cfg: RunConfig, rd: RunDir) -> dict:
    args = trainer_args(cfg, rd, "adamw_torch")
    if "learning_rate" not in cfg.hparams:
        args["learning_rate"] = 1e-5
    args["label_names"] = ["labels"]
    args["prediction_loss_only"] = True
    args["dataloader_num_workers"] = dataloader_workers()
    args["gradient_checkpointing"] = bool(cfg.hparams.get("gradient_checkpointing", True))
    args["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return args


# ---------------------------------------------------------------- train

def train_parler(rd: RunDir, cfg: RunConfig, rep) -> int:
    from transformers import TrainingArguments

    from app.core.backend_kit.hf import MetaContractCallback

    stop = StopFlag().install()
    model = _load(_start_path(cfg, rd))
    freeze_encoders(model, bool(cfg.hparams.get("freeze_text_encoder", True)))
    ptok, dtok = load_tokenizers(cfg.base_model)
    train_ds = _load_concat(_split_dirs(rd, ["train"]))
    val_ds = _load_concat(_split_dirs(rd, ["val"]))
    total = total_steps(cfg, rd)
    collator = ParlerCollator(ptok, dtok, int(model.config.decoder.eos_token_id))
    fe = _feature_extractor(cfg)

    cls = trainer_cls()
    trainer = cls(
        model=model, args=TrainingArguments(**parler_trainer_args(cfg, rd)),
        train_dataset=train_ds, eval_dataset=val_ds, data_collator=collator,
        callbacks=[MetaContractCallback(rep, stop, NanGuard(), total, lambda d, step: write_ckpt_meta(d, cfg, step))],
        **processor_kwarg(cls, fe),
    )
    last = latest_checkpoint(rd) if cfg.resume else None
    trainer.train(resume_from_checkpoint=str(last) if last else None)
    if stop.requested:
        return 143  # the callback already saved a complete checkpoint; resume continues from it

    final = rd.checkpoints_dir / "final"
    trainer.save_model(str(final))
    write_ckpt_meta(final, cfg, trainer.state.global_step)  # meta before .complete
    mark_complete(final)
    return 0


# ---------------------------------------------------------------- eval

def make_synthesizer(model, ptok, dtok, template: str, device="cuda"):
    def synthesize(text: str, row: dict):
        import torch

        d = dtok(describe(row, template), return_tensors="pt").to(device)
        p = ptok(text, return_tensors="pt").to(device)
        torch.manual_seed(SYNTH_SEED)
        with torch.no_grad():
            wav = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                 prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
        return wav.float().cpu().numpy().squeeze(), int(getattr(model.config, "sampling_rate", SAMPLE_RATE))

    return synthesize


def _eval_loss(model, cfg: RunConfig, rd: RunDir, ptok, dtok) -> float:
    from transformers import TrainingArguments

    dirs = [f.with_suffix("") for f in eval_loss_files(rd, cfg)]
    ds = _load_concat(dirs)
    args = TrainingArguments(output_dir=str(rd.path / "work" / "eval_tmp"), per_device_eval_batch_size=
                             int(cfg.hparams.get("batch_size", 2)), bf16=True, report_to=[],
                             remove_unused_columns=False, label_names=["labels"], prediction_loss_only=True,
                             seed=cfg.seed)
    collator = ParlerCollator(ptok, dtok, int(model.config.decoder.eos_token_id))
    trainer = trainer_cls()(model=model, args=args, eval_dataset=ds, data_collator=collator)
    return float(trainer.evaluate()["eval_loss"])


def eval_parler(rd: RunDir, cfg: RunConfig, rep) -> int:
    final = _require_final(rd)
    do_baseline, max_samples = eval_opts(cfg)
    rows = eval_rows(cfg)
    cap = 20 if max_samples is None else min(20, int(max_samples))
    template = template_of(cfg)
    ptok, dtok = load_tokenizers(cfg.base_model)

    model = _load(str(final), bf16=True).to("cuda").eval()
    loss = _eval_loss(model, cfg, rd, ptok, dtok)
    tuned = run_tts_eval(rd, rep, rows, make_synthesizer(model, ptok, dtok, template),
                         loss=loss, prefix="", max_samples=cap)
    del model
    free_gpu()

    base: dict[str, float] = {}
    if do_baseline:
        ds, _ = eval_ds(cfg)

        def compute() -> dict[str, float]:
            m = _load(_start_path(cfg, rd), bf16=True).to("cuda").eval()  # init checkpoint when set, else base
            try:
                res = run_tts_eval(rd, rep, rows, make_synthesizer(m, ptok, dtok, template),
                                   loss=None, prefix="base/", max_samples=cap)
                return {k.removeprefix("base/"): v for k, v in res.items()}  # baseline_metrics re-adds `base/`
            finally:
                del m
                free_gpu()

        base = baseline_metrics(cfg, ds, asr_baseline_key(cfg, max_samples), compute)
    merge_results(rd, tuned, base)
    return 0


# ---------------------------------------------------------------- export

def parler_exporters(rd: RunDir, cfg: RunConfig) -> dict:
    def safetensors(out: Path):
        final = _require_final(rd)
        model = _load(str(final))
        model.save_pretrained(str(out), safe_serialization=True)
        ptok, dtok = load_tokenizers(cfg.base_model)
        ptok.save_pretrained(str(out))
        dtok.save_pretrained(str(out / "description_tokenizer"))
        _feature_extractor(cfg).save_pretrained(str(out))
        return [dir_artifact(rd, "safetensors", out, "transformers")]

    return {"safetensors": safetensors}


def export_parler(rd: RunDir, cfg: RunConfig, rep) -> int:
    return export_all(rd, cfg, rep, parler_exporters(rd, cfg))


train = train_parler
eval = eval_parler  # noqa: A001  (phase handler name looked up by phases._dispatch)
export = export_parler
