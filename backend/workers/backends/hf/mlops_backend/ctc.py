"""CTC ASR for the hf backend: wav2vec2 full fine-tune (`ctc`) and MMS adapter training (`ctc-mms`).

torch and transformers are imported inside functions only. Run plumbing (callbacks, init meta, export, eval
harness) is shared with whisper_full via app.core.backend_kit.

Warm start: wav2vec2 starts from `<init>/vocab.json` + the init model (hf-full); MMS loads the base model,
`init_adapter_layers()` and `adapter.ben.safetensors` (hf-mms-adapter). rd.init_dir is read-only.
"""
from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal

from app.core.backend_kit import (
    NanGuard, StopFlag, eval_opts, export_all, init_source, latest_checkpoint, mark_complete, max_skip_share,
    write_init_meta,
)
from app.core.backend_kit.audio_refs import SAMPLE_RATE, check_audio_refs
from app.core.backend_kit.runutil import (
    cap_val, dir_artifact, eval_ds, eval_rows, free_gpu, stopped_early, total_steps, trainer_args, train_ref,
)
from app.core.backend_kit.whisper import asr_baseline_key, dataloader_workers, processor_kwarg
from app.core.contract import RunConfig, RunDir, write_prepared_refs
from app.core.evaluation import baseline_metrics, merge_results, run_asr_eval
from app.core.prepare import PrepareError, ensure_prepared, iter_rows

from .models import MODELS
from .prepare.ctc import (
    build_ctc, build_ctc_vocab, ctc_greedy_decode, ctc_labels, ctc_spec, mms_vocab, pad_ctc_labels,
    first_last_lines, special_ids,
)

BACKEND = "hf"
MMS_BASE = "facebook/mms-1b-all"
ADAPTER_FILE = "adapter.ben.safetensors"
InitPlan = Literal["base", "full_weights", "adapter"]


def is_mms(cfg: RunConfig) -> bool:
    info = MODELS.get(cfg.base_model)
    return info is not None and info.family == "ctc-mms"


def plan_ctc_init(meta: dict | None, mms: bool) -> InitPlan:
    """Starting model from an init checkpoint's meta: wav2vec2 continues from full weights, MMS from an adapter."""
    if meta is None:
        return "base"
    fmt = meta.get("format")
    want = "hf-mms-adapter" if mms else "hf-full"
    if fmt != want:
        raise ValueError(f"unsupported init checkpoint format '{fmt}' for a {'MMS adapter' if mms else 'wav2vec2'} "
                         f"run on backend {BACKEND}: only {want} checkpoints can be used")
    return "adapter" if mms else "full_weights"


def write_ckpt_meta(ckpt_dir: Path, cfg: RunConfig, step: int) -> None:
    mms = is_mms(cfg)
    write_init_meta(ckpt_dir, backend=BACKEND, base_model=cfg.base_model, method="adapter" if mms else "full",
                    format="hf-mms-adapter" if mms else "hf-full", step=step)


# ---------------------------------------------------------------- prepare

def resolve_vocab(rd: RunDir, cfg: RunConfig, texts: Callable[[], Iterable[str]],
                  mms_loader: Callable[[], dict[str, int]]) -> tuple[dict[str, int], Literal["new", "fixed"]]:
    """Vocab and build mode. Warm start: the source checkpoint's vocab.json, fixed (spec 14.6). MMS: the `ben`
    vocabulary of the base model, fixed. Otherwise built from the training texts (one pass)."""
    choice = init_source(cfg.model_copy(update={"resume": False}), rd)
    if choice is not None and choice.kind == "init":
        plan_ctc_init(choice.meta, is_mms(cfg))
        vp = choice.path / "vocab.json"
        if not vp.is_file():
            raise PrepareError(f"init checkpoint {choice.path} has no vocab.json")
        return json.loads(vp.read_text(encoding="utf-8")), "fixed"
    if is_mms(cfg):
        return mms_loader(), "fixed"
    return build_ctc_vocab(texts()), "new"


def prepare_ctc(rd: RunDir, cfg: RunConfig, rep) -> int:
    train_ds = cfg.datasets.get("train")

    def texts():
        return (r["text"] for r in iter_rows(train_ds, "train"))

    def load_mms():
        from transformers import Wav2Vec2Processor

        return mms_vocab(Wav2Vec2Processor.from_pretrained(cfg.base_model))

    vocab, mode = resolve_vocab(rd, cfg, texts, load_mms)
    spec = ctc_spec(cfg, vocab)
    share = max_skip_share(cfg)
    refs = []
    for role, key, splits in (("train", "train", ["train", "val"]), ("eval", "eval", None)):
        ds = cfg.datasets.get(key)
        if ds is None:
            continue
        # The eval set is never filtered by the vocab: unseen characters only cost WER.
        build = build_ctc(vocab, mode if role == "train" else "new", rep)
        pi = ensure_prepared(cfg, ds, spec, build, splits=splits, max_skip_share=share)
        refs.append({"role": role, **pi.model_dump()})
    write_prepared_refs(rd, refs)
    return 0


prepare = prepare_ctc  # phase handler looked up by phases._dispatch


# ---------------------------------------------------------------- data

def _vocab_path(rd: RunDir) -> Path:
    return Path(train_ref(rd)["path"]) / "vocab.json"


def read_vocab(rd: RunDir) -> dict[str, int]:
    return json.loads(_vocab_path(rd).read_text(encoding="utf-8"))


class RowsDataset:
    """Prepared rows in memory as compact (audio_path, duration_s, text) tuples; audio is decoded by the collator."""

    def __init__(self, rows: list[tuple[str, float, str]]):
        self.rows = rows

    @classmethod
    def from_jsonl(cls, path: Path) -> "RowsDataset":
        rows = []
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows.append((r["audio_path"], float(r["duration_s"]), r["text"]))
        return cls(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        path, dur, text = self.rows[i]
        return {"audio_path": path, "duration_s": dur, "text": text}

    def select(self, indices) -> "RowsDataset":
        return RowsDataset([self.rows[i] for i in indices])


def check_prepared_refs(rd: RunDir) -> None:
    """First and last audio_path of train.jsonl must still exist (reads two lines, not the file)."""
    prepared = Path(train_ref(rd)["path"])
    ends = first_last_lines(prepared / "train.jsonl")
    if ends:
        check_audio_refs([r["audio_path"] for r in ends], str(prepared))


def load_train_val(rd: RunDir) -> tuple[RowsDataset, RowsDataset]:
    prepared = Path(train_ref(rd)["path"])
    check_prepared_refs(rd)
    return RowsDataset.from_jsonl(prepared / "train.jsonl"), RowsDataset.from_jsonl(prepared / "val.jsonl")


def _default_loader(path):
    from app.core.backend_kit.audio import load_audio

    return load_audio(path, SAMPLE_RATE)


class CtcCollator:
    """Loads and resamples audio per batch, pads inputs, tokenizes text by vocab lookup, pads labels with -100."""

    def __init__(self, feature_extractor, vocab: dict[str, int], loader: Callable = _default_loader):
        self.fe = feature_extractor
        self.vocab = vocab
        self.loader = loader

    def __call__(self, features: list[dict]) -> dict:
        import torch

        audios = [self.loader(str(f["audio_path"])) for f in features]
        b = self.fe(audios, sampling_rate=SAMPLE_RATE, padding=True, return_attention_mask=True,
                    return_tensors="pt")
        labels = pad_ctc_labels([ctc_labels(f["text"], self.vocab) for f in features])
        return {"input_values": b.input_values, "attention_mask": b.attention_mask,
                "labels": torch.tensor(labels, dtype=torch.long)}


# ---------------------------------------------------------------- model

def _feature_extractor(cfg: RunConfig):
    from transformers import Wav2Vec2FeatureExtractor

    return Wav2Vec2FeatureExtractor.from_pretrained(cfg.base_model)


def save_processor(cfg: RunConfig, vocab_path: Path, out: Path) -> None:
    """Feature extractor + CTC tokenizer (built from the run's vocab.json) + the vocab file itself."""
    from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor

    vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    unk_id, pad_id = special_ids(vocab)
    inv = {v: k for k, v in vocab.items()}
    tok = Wav2Vec2CTCTokenizer(str(vocab_path), unk_token=inv[unk_id], pad_token=inv[pad_id],
                               word_delimiter_token="|", bos_token=None, eos_token=None)
    Wav2Vec2Processor(feature_extractor=_feature_extractor(cfg), tokenizer=tok).save_pretrained(str(out))


def adapter_state(model) -> dict:
    """Adapter layers + lm_head as contiguous CPU tensors, ready for safetensors."""
    return {k: v.detach().cpu().contiguous().clone() for k, v in model._get_adapters().items()}


def save_adapter(model, out_dir: Path) -> None:
    from safetensors.torch import save_file

    save_file(adapter_state(model), str(Path(out_dir) / ADAPTER_FILE))


def _ctc_kwargs(vocab: dict[str, int]) -> dict:
    _, pad_id = special_ids(vocab)
    return {"vocab_size": len(vocab), "pad_token_id": pad_id, "ctc_loss_reduction": "mean",
            "ctc_zero_infinity": True, "layerdrop": 0.0}


def check_adapter_keys(expected, loaded) -> None:
    """Raise naming the keys when an adapter file does not match the adapter parameters of the model."""
    exp, got = set(expected), set(loaded)
    if exp != got:
        raise RuntimeError(f"adapter checkpoint does not match the model: missing {sorted(exp - got)}, "
                           f"unexpected {sorted(got - exp)}")


def load_adapter_into(model, path: Path) -> None:
    """Load adapter.ben.safetensors into a model that has adapter layers; keys must match exactly."""
    from safetensors.torch import load_file

    state = load_file(str(path))
    check_adapter_keys(model._get_adapters().keys(), state.keys())
    model.load_state_dict(state, strict=False)  # only adapter keys are given; they were checked above


def build_mms(cfg: RunConfig, vocab: dict[str, int], adapter: Path | None = None, *, fresh: bool = False):
    """MMS base + `ben` target head. `adapter`: fresh adapter layers then that adapter file (warm start, eval,
    export). `fresh`: new adapter layers (training from scratch). Neither: the pretrained `ben` adapter."""
    from transformers import Wav2Vec2ForCTC

    model = Wav2Vec2ForCTC.from_pretrained(cfg.base_model, target_lang="ben", ignore_mismatched_sizes=True,
                                           **_ctc_kwargs(vocab))
    if adapter is not None or fresh:
        model.init_adapter_layers()
    if adapter is not None:
        load_adapter_into(model, adapter)
    return model


def new_model(cfg: RunConfig, rd: RunDir, vocab: dict[str, int], *, train: bool):
    """Model at the start of this run (init checkpoint when set, else the base model).
    `train=True` prepares it for training; False gives the starting point used for the baseline."""
    import torch
    from transformers import Wav2Vec2ForCTC

    mms = is_mms(cfg)
    choice = init_source(cfg.model_copy(update={"resume": False}), rd)
    plan = plan_ctc_init(choice.meta, mms) if choice is not None else "base"
    if mms:
        adapter = choice.path / ADAPTER_FILE if plan == "adapter" else None
        model = build_mms(cfg, vocab, adapter, fresh=train and adapter is None)
        if train:
            freeze_to_adapter(model)
    else:
        src = str(choice.path) if plan == "full_weights" else cfg.base_model
        model = Wav2Vec2ForCTC.from_pretrained(src, **_ctc_kwargs(vocab))
        if train:
            model.freeze_feature_encoder()
    if not train:
        model.to(torch.bfloat16)
    return model


def checkpoint_files(mms: bool) -> list[str]:
    """Files every complete checkpoint of a family holds beyond Trainer state: MMS is adapter-only (R20)."""
    common = ["config.json", "vocab.json", "init_meta.json", ".complete"]
    return common + (["adapter.ben.safetensors"] if mms else ["model.safetensors"])


def freeze_to_adapter(model) -> None:
    model.freeze_base_model()
    for p in model._get_adapters().values():
        p.requires_grad = True


def adapter_param_count(model) -> int:
    return sum(p.numel() for p in model._get_adapters().values())


def trainable_param_count(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_final_model(cfg: RunConfig, rd: RunDir, final: Path, *, bf16: bool = False):
    """The trained model from a complete checkpoint: full weights (wav2vec2) or base + adapter (MMS)."""
    import torch
    from transformers import Wav2Vec2ForCTC

    if is_mms(cfg):
        model = build_mms(cfg, read_vocab(rd), final / ADAPTER_FILE)
        return model.to(torch.bfloat16) if bf16 else model
    kw = {"torch_dtype": torch.bfloat16} if bf16 else {}
    return Wav2Vec2ForCTC.from_pretrained(str(final), **kw)


def adapter_trainer_cls():
    """Trainer whose checkpoints hold only config + adapter (the MMS base is 1B params; R20)."""
    import os

    from transformers import Trainer

    class AdapterTrainer(Trainer):
        def _save(self, output_dir=None, state_dict=None):
            out = output_dir or self.args.output_dir
            os.makedirs(out, exist_ok=True)
            self.model.config.save_pretrained(out)
            save_adapter(self.model, Path(out))

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            load_adapter_into(self.model if model is None else model, Path(resume_from_checkpoint) / ADAPTER_FILE)

    return AdapterTrainer


# ---------------------------------------------------------------- train

def ctc_trainer_args(cfg: RunConfig, rd: RunDir) -> dict:
    args = trainer_args(cfg, rd, "adamw_torch")
    if "learning_rate" not in cfg.hparams:
        args["learning_rate"] = 1e-3 if is_mms(cfg) else 1e-4
    args["label_names"] = ["labels"]
    args["prediction_loss_only"] = True
    args["dataloader_num_workers"] = dataloader_workers()
    args["gradient_checkpointing"] = bool(cfg.hparams.get("gradient_checkpointing", True))
    args["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return args


def train_ctc(rd: RunDir, cfg: RunConfig, rep) -> int:
    from transformers import Trainer, TrainingArguments

    from app.core.backend_kit.hf import MetaContractCallback

    stop = StopFlag().install()
    vocab = read_vocab(rd)
    model = new_model(cfg, rd, vocab, train=True)
    train_ds, val_ds = load_train_val(rd)
    val_ds = cap_val(val_ds, cfg)  # eval every save_steps: a seeded subset, not the whole val split
    total = total_steps(cfg, rd)
    mms = is_mms(cfg)

    def write_meta(d: Path, step: int) -> None:
        shutil.copyfile(_vocab_path(rd), d / "vocab.json")
        write_ckpt_meta(d, cfg, step)

    fe = _feature_extractor(cfg)
    trainer = (adapter_trainer_cls() if mms else Trainer)(
        model=model, args=TrainingArguments(**ctc_trainer_args(cfg, rd)),
        train_dataset=train_ds, eval_dataset=val_ds, data_collator=CtcCollator(fe, vocab),
        callbacks=[MetaContractCallback(rep, stop, NanGuard(), total, write_meta)],
        **processor_kwarg(Trainer, fe),
    )
    last = latest_checkpoint(rd) if cfg.resume else None
    trainer.train(resume_from_checkpoint=str(last) if last else None)
    if stopped_early(stop, trainer):
        return 143  # the callback already saved a complete checkpoint; resume continues from it

    final = rd.checkpoints_dir / "final"
    trainer.save_model(str(final))  # MMS: config + adapter only
    save_processor(cfg, _vocab_path(rd), final)
    shutil.copyfile(_vocab_path(rd), final / "vocab.json")
    write_ckpt_meta(final, cfg, trainer.state.global_step)  # meta before .complete
    mark_complete(final)
    return 0


# ---------------------------------------------------------------- eval

def make_ctc_transcriber(model, feature_extractor, vocab: dict[str, int], device) -> Callable[[list[str]], list[str]]:
    inv = {v: k for k, v in vocab.items()}
    unk_id, pad_id = special_ids(vocab)

    def transcribe(paths: list[str]) -> list[str]:
        import torch

        audios = [_default_loader(p) for p in paths]
        b = feature_extractor(audios, sampling_rate=SAMPLE_RATE, padding=True, return_attention_mask=True,
                              return_tensors="pt")
        with torch.no_grad():
            logits = model(input_values=b.input_values.to(device=device, dtype=model.dtype),
                           attention_mask=b.attention_mask.to(device)).logits
        return [ctc_greedy_decode(r, inv, pad_id, unk_id) for r in logits.argmax(-1).cpu().tolist()]

    return transcribe


def _require_final(rd: RunDir) -> Path:
    final = rd.checkpoints_dir / "final"
    if not (final / ".complete").exists():
        raise RuntimeError(f"no completed final checkpoint at {final}")
    return final


def eval_ctc(rd: RunDir, cfg: RunConfig, rep) -> int:
    final = _require_final(rd)
    do_baseline, max_samples = eval_opts(cfg)
    rows = eval_rows(cfg)
    check_prepared_refs(rd)
    vocab = read_vocab(rd)
    fe = _feature_extractor(cfg)

    def run(model, prefix: str) -> dict[str, float]:
        model.to("cuda").eval()
        return run_asr_eval(rd, rep, rows, make_ctc_transcriber(model, fe, vocab, "cuda"),
                            prefix=prefix, max_samples=max_samples)

    model = load_final_model(cfg, rd, final, bf16=True)
    tuned = run(model, "")
    del model
    free_gpu()

    base: dict[str, float] = {}
    if do_baseline:
        ds, _ = eval_ds(cfg)

        def compute() -> dict[str, float]:
            m = new_model(cfg, rd, vocab, train=False)  # init checkpoint when set, else the base model
            try:
                res = run(m, "base/")
                return {k.removeprefix("base/"): v for k, v in res.items()}  # baseline_metrics re-adds `base/`
            finally:
                del m
                free_gpu()

        base = baseline_metrics(cfg, ds, asr_baseline_key(cfg, max_samples), compute)
    merge_results(rd, tuned, base)
    return 0


# ---------------------------------------------------------------- export

def ctc_exporters(rd: RunDir, cfg: RunConfig) -> dict:
    """safetensors for both families; adapter-safetensors for MMS only. ct2 is never offered (R19)."""
    def safetensors(out: Path):
        model = load_final_model(cfg, rd, _require_final(rd))  # MMS: rebuilt from base + adapter
        model.save_pretrained(str(out), safe_serialization=True)
        save_processor(cfg, _vocab_path(rd), out)
        return [dir_artifact(rd, "safetensors", out, "transformers")]

    def adapter(out: Path):
        final = _require_final(rd)
        shutil.copyfile(final / ADAPTER_FILE, out / ADAPTER_FILE)
        shutil.copyfile(_vocab_path(rd), out / "vocab.json")
        return [dir_artifact(rd, "adapter-safetensors", out, "transformers")]

    fns = {"safetensors": safetensors}
    if is_mms(cfg):
        fns["adapter-safetensors"] = adapter
    return fns


def export_ctc(rd: RunDir, cfg: RunConfig, rep) -> int:
    return export_all(rd, cfg, rep, ctc_exporters(rd, cfg))


train = train_ctc
eval = eval_ctc  # noqa: A001  (phase handler name looked up by phases._dispatch)
export = export_ctc
