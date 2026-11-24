"""Whisper ASR dataset converter and batched transcriber, shared by the unsloth and hf backends.

transformers, datasets, numpy and torch are imported inside functions only, so importing this module
stays light. The package __init__ does not import it.
"""
from __future__ import annotations

import shutil
from collections.abc import Callable, Iterator
from pathlib import Path

from app.core.prepare import BuildResult, PrepSpec, Skip, ensure_prepared
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
        "storage": "audio-int16",
    })


def row_to_example(row: dict, *, load: Callable, encode: Callable, tokenize: Callable,
                   sr: int = SAMPLE_RATE, max_label_tokens: int = MAX_LABEL_TOKENS) -> dict | str:
    """One manifest row -> {"audio", "labels"}, or the skip reason (a str)."""
    try:
        audio = load(row["audio_path"])
    except Exception as e:  # noqa: BLE001 - any decode failure skips the row (spec section 10)
        return f"undecodable audio: {type(e).__name__}"
    seconds = len(audio) / sr
    if seconds > MAX_SECONDS:
        return f"duration {seconds:.1f} s > {MAX_SECONDS} s"
    labels = list(tokenize(bn_normalize(row["text"], mode="asr")))
    if len(labels) > max_label_tokens:
        return f"{len(labels)} label tokens > {max_label_tokens}"
    return {"audio": encode(audio), "labels": labels}


def pad_labels(label_lists: list[list[int]], start_id: int) -> list[list[int]]:
    """Strip the leading decoder-start token when every row has it (the model prepends it), pad with -100."""
    rows = [list(x) for x in label_lists]
    if rows and all(r and r[0] == start_id for r in rows):
        rows = [r[1:] for r in rows]
    width = max((len(r) for r in rows), default=0)
    return [r + [IGNORE] * (width - len(r)) for r in rows]


def to_int16(audio):
    """float [-1, 1] -> int16 (clipped, *32767)."""
    import numpy as np

    return (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767.0).astype(np.int16)


def int16_to_float(audio):
    import numpy as np

    return np.asarray(audio, dtype=np.int16).astype(np.float32) / 32768.0


def batch_parts(features: list[dict], start_id: int) -> tuple[list, list[list[int]]]:
    """Collator logic without torch: float32 audios and padded labels."""
    return ([int16_to_float(f["audio"]) for f in features],
            pad_labels([[int(t) for t in f["labels"]] for f in features], start_id))


class WhisperCollator:
    """Batches prepared rows (int16 audio + labels): mel features computed on the fly by the processor."""

    def __init__(self, processor, decoder_start_id: int | None = None):
        self.processor = processor
        self.start_id = (decoder_start_id if decoder_start_id is not None
                         else processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>"))

    def __call__(self, features: list[dict]) -> dict:
        import torch

        audios, labels = batch_parts(features, self.start_id)
        feats = self.processor.feature_extractor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        return {"input_features": feats.input_features, "labels": torch.tensor(labels, dtype=torch.long)}


def _make_fns(processor, sr: int):
    from app.core.backend_kit.audio import load_audio

    tok = processor.tokenizer
    tok.set_prefix_tokens(language=LANGUAGE, task=TASK)

    def load(path):
        return load_audio(path, sr)

    def tokenize(text):
        return tok(text).input_ids

    return load, to_int16, tokenize


def iter_examples(items: list[dict], load, encode, tokenize) -> Iterator[dict]:
    """Rows -> dataset records. A skipped row yields empty audio/labels and its reason in `skip`
    (so it survives worker processes); the caller drops those after generation."""
    for row in items:
        ex = row_to_example(row, load=load, encode=encode, tokenize=tokenize)
        sid = str(row.get("sample_id", ""))
        if isinstance(ex, str):
            yield {"sample_id": sid, "audio": [], "labels": [], "skip": ex}
        else:
            yield {"sample_id": sid, **ex, "skip": ""}


# Fork-inherited state for from_generator workers (rows are never pickled or hashed).
_CTX: dict[str, tuple] = {}


def _gen(ranges, key):
    items, fns = _CTX[key]
    for lo, hi in ranges:
        yield from iter_examples(items[lo:hi], *fns)


def _write_split(items: list[dict], out_dir: Path, ctx: tuple) -> tuple[int, list[Skip]]:
    """Stream examples into an HF dataset at out_dir (parallel, never all in memory).
    Returns (kept, skips)."""
    import os

    from datasets import Dataset, Features, Sequence, Value

    split = out_dir.name
    feats = Features({"sample_id": Value("string"), "audio": Sequence(Value("int16")),
                      "labels": Sequence(Value("int64")), "skip": Value("string")})
    procs = max(1, min(8, os.cpu_count() or 1))
    chunk = max(1, len(items) // (procs * 16) or 1)
    ranges = [(i, min(i + chunk, len(items))) for i in range(0, len(items), chunk)]
    key = f"{os.getpid()}:{id(items)}"
    _CTX[key] = (items, ctx)
    cache = out_dir.with_name(f".{split}.hf-cache")  # fresh per build: no cached generator reuse
    shutil.rmtree(cache, ignore_errors=True)
    try:
        ds = Dataset.from_generator(
            _gen, features=feats, cache_dir=str(cache), gen_kwargs={"ranges": ranges, "key": key},
            num_proc=procs if procs > 1 and len(ranges) >= procs else None,
            writer_batch_size=32)  # small write batches: rows can be ~1 MB
        skip_col = ds["skip"]
        ids = ds["sample_id"]
        skips = [Skip(sample_id=i, split=split, reason=r) for i, r in zip(ids, skip_col) if r]
        keep = [i for i, r in enumerate(skip_col) if not r]
        ds.select(keep).save_to_disk(str(out_dir))  # save_to_disk flattens: one more pass, no extra filter copy
    finally:
        _CTX.pop(key, None)
        shutil.rmtree(cache, ignore_errors=True)
    return len(keep), skips


def build_whisper(processor) -> Callable[[dict[str, list[dict]], Path], BuildResult]:
    def build(rows: dict[str, list[dict]], out: Path) -> BuildResult:
        ctx = _make_fns(processor, SAMPLE_RATE)
        counts: dict[str, int] = {}
        skipped: list[Skip] = []
        for split, items in rows.items():
            counts[split], skips = _write_split(items, out / split, ctx)
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


def prepare_whisper(rd, cfg, rep) -> int:
    from app.core.backend_kit import max_skip_share
    from app.core.contract import write_prepared_refs

    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained(cfg.base_model)
    spec = whisper_spec(cfg)
    build = build_whisper(processor)
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
           "prepare_whisper", "WhisperCollator", "to_int16", "int16_to_float", "batch_parts"]
