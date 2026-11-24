"""Whisper ASR dataset converter and batched transcriber, shared by the unsloth and hf backends.

transformers, datasets, numpy and torch are imported inside functions only, so importing this module
stays light. The package __init__ does not import it.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

from app.core.backend_kit.audio_refs import (  # noqa: F401 - re-exported for callers
    PROBE_THREADS, PROGRESS_EVERY, check_audio_refs, probe_duration, probed_rows, progress_ticker,
)
from app.core.prepare import BuildResult, PrepSpec, Skip, empty_split_error, ensure_prepared
from app.core.text import BN_NORMALIZE_VERSION, bn_normalize

SAMPLE_RATE = 16000
MAX_SECONDS = 30
MAX_LABEL_TOKENS = 448
IGNORE = -100
LANGUAGE = "bengali"
TASK = "transcribe"


def whisper_spec(cfg) -> "PrepSpec":
    return PrepSpec(family="whisper", version=1, params={
        "base_model": cfg.base_model,
        "sample_rate": SAMPLE_RATE,
        "bn_normalize_version": BN_NORMALIZE_VERSION,
        "max_label_tokens": MAX_LABEL_TOKENS,
        "storage": "audio-ref",
        "backend": cfg.backend,  # unsloth and hf pin different datasets/transformers: never share a cache
    })


def row_to_example(row: dict, *, probe: Callable, tokenize: Callable,
                   max_label_tokens: int = MAX_LABEL_TOKENS) -> dict | str:
    """One manifest row -> prepared record (no audio, only its path), or the skip reason (a str)."""
    try:
        probed = probe(row["audio_path"])
    except Exception as e:  # noqa: BLE001 - any decode failure skips the row (spec section 10)
        return f"undecodable audio: {type(e).__name__}"
    given = row.get("duration_s")
    seconds = float(given) if given else float(probed)
    if seconds > MAX_SECONDS:
        return f"duration {seconds:.1f} s > {MAX_SECONDS} s"
    labels = [int(t) for t in tokenize(bn_normalize(row["text"], mode="asr"))]
    if len(labels) > max_label_tokens:
        return f"{len(labels)} label tokens > {max_label_tokens}"
    return {"sample_id": str(row.get("sample_id", "")), "audio_path": str(row["audio_path"]),
            "duration_s": seconds, "labels": labels}


def pad_labels(label_lists: list[list[int]], start_id: int) -> list[list[int]]:
    """Strip the leading decoder-start token when every row has it (the model prepends it), pad with -100."""
    rows = [list(x) for x in label_lists]
    if rows and all(r and r[0] == start_id for r in rows):
        rows = [r[1:] for r in rows]
    width = max((len(r) for r in rows), default=0)
    return [r + [IGNORE] * (width - len(r)) for r in rows]


def _default_loader(path):
    from app.core.backend_kit.audio import load_audio

    return load_audio(path, SAMPLE_RATE)


def batch_parts(features: list[dict], start_id: int, loader: Callable = _default_loader) -> tuple[list, list[list[int]]]:
    """Collator logic without torch: loaded float32 audios and padded labels."""
    return ([loader(str(f["audio_path"])) for f in features],
            pad_labels([[int(t) for t in f["labels"]] for f in features], start_id))


class WhisperCollator:
    """Batches prepared rows (audio_path + labels): audio is loaded and mel features computed on the fly."""

    def __init__(self, processor, decoder_start_id: int | None = None, loader: Callable = _default_loader):
        self.processor = processor
        self.loader = loader
        self.start_id = (decoder_start_id if decoder_start_id is not None
                         else processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>"))

    def __call__(self, features: list[dict]) -> dict:
        import torch

        audios, labels = batch_parts(features, self.start_id, self.loader)
        feats = self.processor.feature_extractor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        return {"input_features": feats.input_features, "labels": torch.tensor(labels, dtype=torch.long)}


def _write_split(items: list[dict], out_dir: Path, probe: Callable, tokenize: Callable,
                 tick: Callable[[int], None] = lambda n: None) -> tuple[int, list[Skip]]:
    """Metadata-only dataset (tens of MB for ~1M clips): files are probed by a thread pool (row order kept),
    columns accumulated in memory."""
    cols: dict[str, list] = {"sample_id": [], "audio_path": [], "duration_s": [], "labels": []}
    skips: list[Skip] = []
    for row, res in probed_rows(items, probe, tick):
        def cached(_path, res=res):
            if isinstance(res, Exception):
                raise res
            return res

        ex = row_to_example(row, probe=cached, tokenize=tokenize)
        if isinstance(ex, str):
            skips.append(Skip(sample_id=str(row.get("sample_id", "")), split=out_dir.name, reason=ex))
            continue
        for k in cols:
            cols[k].append(ex[k])
    if not cols["sample_id"]:
        raise empty_split_error(out_dir.name, skips)
    from datasets import Dataset

    Dataset.from_dict(cols).save_to_disk(str(out_dir))
    return len(cols["sample_id"]), skips


def build_whisper(processor, rep=None) -> Callable[[dict[str, list[dict]], Path], BuildResult]:
    def build(rows: dict[str, list[dict]], out: Path) -> BuildResult:
        tok = processor.tokenizer
        tok.set_prefix_tokens(language=LANGUAGE, task=TASK)
        tick = progress_ticker(rep, sum(len(v) for v in rows.values()))

        counts: dict[str, int] = {}
        skipped: list[Skip] = []
        for split, items in rows.items():
            counts[split], skips = _write_split(items, out / split, probe_duration,
                                                lambda t: tok(t).input_ids, tick)
            skipped.extend(skips)
        return BuildResult(counts=counts, skipped=skipped, files=list(rows))

    return build


def make_transcriber(model, processor, device, batch_max_new_tokens: int = 225) -> Callable[[list[str]], list[str]]:
    def transcribe(paths: list[str]) -> list[str]:
        import torch

        from app.core.backend_kit.audio import load_audio

        audios = [load_audio(p, SAMPLE_RATE) for p in paths]
        feats = processor.feature_extractor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        feats = feats.to(device=device, dtype=getattr(model, "dtype", None) or feats.dtype)
        with torch.no_grad():
            ids = model.generate(input_features=feats, language=LANGUAGE, task=TASK,
                                 max_new_tokens=batch_max_new_tokens)
        return [t.strip() for t in processor.batch_decode(ids, skip_special_tokens=True)]

    return transcribe


def check_prepared_audio(ds, prepared) -> None:
    """First and last audio_path of a prepared split must still exist."""
    if len(ds):
        check_audio_refs([ds[0]["audio_path"], ds[len(ds) - 1]["audio_path"]], str(prepared))


def check_prepared_refs(rd) -> None:
    from datasets import load_from_disk

    from app.core.backend_kit.runutil import train_ref

    prepared = Path(train_ref(rd)["path"])
    check_prepared_audio(load_from_disk(str(prepared / "train")), prepared)


def load_train_val(rd):
    """(train, val, prepared dir) of the prepared whisper dataset; audio references are checked."""
    from datasets import load_from_disk

    from app.core.backend_kit.runutil import train_ref

    prepared = Path(train_ref(rd)["path"])
    train_ds = load_from_disk(str(prepared / "train"))
    check_prepared_audio(train_ds, prepared)
    return train_ds, load_from_disk(str(prepared / "val")), prepared


def load_processor(base_model: str):
    from transformers import WhisperProcessor

    p = WhisperProcessor.from_pretrained(base_model)
    p.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)
    return p


def processor_kwarg(trainer_cls, component) -> dict:
    """Seq2SeqTrainer kwarg for the feature extractor: `processing_class` (newer transformers) or `tokenizer`."""
    import inspect

    name = "processing_class" if "processing_class" in inspect.signature(trainer_cls.__init__).parameters else "tokenizer"
    return {name: component}


def dataloader_workers() -> int:
    return min(8, os.cpu_count() or 1)


def whisper_eval_loss(model, processor, split_dirs, *, batch_size: int, seed: int, work_dir,
                      max_rows: int | None = None) -> float | None:
    """Eval loss over prepared split dirs (collator decodes audio per batch); None when they are all empty.
    With `max_rows`, over a seeded subset of at most that many rows (hparams.eval.max_val_samples)."""
    from datasets import concatenate_datasets, load_from_disk
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    parts = [load_from_disk(str(d)) for d in split_dirs if Path(d).is_dir()]
    parts = [p for p in parts if len(p)]
    if not parts:
        return None
    args = Seq2SeqTrainingArguments(
        output_dir=str(work_dir), per_device_eval_batch_size=batch_size, bf16=True, report_to=[],
        remove_unused_columns=False, label_names=["labels"], seed=seed, dataloader_num_workers=dataloader_workers())
    data = concatenate_datasets(parts)
    if max_rows is not None:
        from app.core.backend_kit.runutil import val_subset_indices

        idx = val_subset_indices(len(data), max_rows, seed)
        data = data if idx is None else data.select(idx)
    trainer = Seq2SeqTrainer(model=model, args=args, eval_dataset=data,
                             data_collator=WhisperCollator(processor, int(model.config.decoder_start_token_id)),
                             **processor_kwarg(Seq2SeqTrainer, processor.feature_extractor))
    return float(trainer.evaluate()["eval_loss"])


def asr_baseline_key(cfg, max_samples: int | None) -> dict:
    from app.core.backend_kit.runutil import max_val_samples
    from app.core.evaluation import ASR_MAX_SAMPLES

    key: dict = {"max_samples": max_samples if max_samples is not None else ASR_MAX_SAMPLES,  # effective cap
                 "max_val_samples": max_val_samples(cfg.hparams),  # eval-loss subset
                 "backend": cfg.backend}  # each backend's loader computes its own base/
    if cfg.init_from is not None:
        key["init"] = [cfg.init_from.source_run, cfg.init_from.checkpoint]
    return key


def ct2_command(merged: Path, out: Path) -> list[str]:
    """ct2-transformers-converter from this interpreter's environment (never a bare PATH lookup)."""
    script = Path(sys.executable).parent / ("ct2-transformers-converter.exe" if sys.platform == "win32"
                                            else "ct2-transformers-converter")
    head = [str(script)] if script.exists() else [sys.executable, "-m", "ctranslate2.converters.transformers"]
    return head + ["--model", str(merged), "--output_dir", str(out), "--force", "--quantization", "float16",
                   "--copy_files", "tokenizer.json", "preprocessor_config.json"]


def ensure_tokenizer_json(model_dir: Path) -> None:
    """ct2 --copy_files needs tokenizer.json. Older transformers (4.46, the hf backend) may save only the slow
    tokenizer files (vocab.json, merges.txt); convert them to the fast tokenizer in place when it is missing."""
    model_dir = Path(model_dir)
    if (model_dir / "tokenizer.json").exists():
        return
    from transformers import WhisperTokenizerFast

    WhisperTokenizerFast.from_pretrained(str(model_dir)).save_pretrained(str(model_dir))


def run_ct2(merged: Path, out: Path) -> None:
    import subprocess

    ensure_tokenizer_json(merged)
    subprocess.run(ct2_command(merged, out), check=True)


def prepare_whisper(rd, cfg, rep) -> int:
    from app.core.backend_kit import max_skip_share
    from app.core.contract import write_prepared_refs

    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained(cfg.base_model)
    spec = whisper_spec(cfg)
    build = build_whisper(processor, rep)
    share = max_skip_share(cfg)
    refs = []
    for role, key, splits in (("train", "train", ["train", "val"]), ("eval", "eval", None)):
        ds = cfg.datasets.get(key)
        if ds is None:
            continue
        pi = ensure_prepared(cfg, ds, spec, build, splits=splits, max_skip_share=share)
        refs.append({"role": role, **pi.model_dump()})
    write_prepared_refs(rd, refs)
    return 0


__all__ = ["whisper_spec", "row_to_example", "pad_labels", "build_whisper", "make_transcriber",
           "prepare_whisper", "WhisperCollator", "batch_parts", "probe_duration", "check_audio_refs",
           "check_prepared_audio", "check_prepared_refs", "load_train_val", "load_processor", "processor_kwarg",
           "dataloader_workers", "whisper_eval_loss", "asr_baseline_key", "ct2_command", "ensure_tokenizer_json", "run_ct2"]
