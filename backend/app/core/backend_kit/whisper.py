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
    })


def row_to_example(row: dict, *, load: Callable, extract: Callable, tokenize: Callable,
                   sr: int = SAMPLE_RATE, max_label_tokens: int = MAX_LABEL_TOKENS) -> dict | str:
    """One manifest row -> {"input_features", "labels"}, or the skip reason (a str)."""
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
    return {"input_features": extract(audio), "labels": labels}


def pad_labels(label_lists: list[list[int]], start_id: int) -> list[list[int]]:
    """Strip the leading decoder-start token when every row has it (the model prepends it), pad with -100."""
    rows = [list(x) for x in label_lists]
    if rows and all(r and r[0] == start_id for r in rows):
        rows = [r[1:] for r in rows]
    width = max((len(r) for r in rows), default=0)
    return [r + [IGNORE] * (width - len(r)) for r in rows]


def _make_fns(processor, sr: int):
    import numpy as np

    from app.core.backend_kit.audio import load_audio

    fe, tok = processor.feature_extractor, processor.tokenizer
    tok.set_prefix_tokens(language=LANGUAGE, task=TASK)

    def load(path):
        return load_audio(path, sr)

    def extract(audio):  # float16 halves the on-disk size; the collator casts back
        return np.asarray(fe(audio, sampling_rate=sr, return_tensors="np").input_features[0], dtype=np.float16)

    def tokenize(text):
        return tok(text).input_ids

    return load, extract, tokenize


def _noop():
    return None


class _RowGen:
    """Generator callable for Dataset.from_generator. Pickles to a constant so `datasets` never hashes
    the (possibly 1M-row) row list when it fingerprints the generator."""

    def __init__(self, it: Iterator[dict]):
        self.it = it

    def __call__(self):
        yield from self.it

    def __reduce__(self):
        return (_noop, ())


def _write_split(gen_rows: Iterator[dict], out_dir: Path, n_mels: int, frames: int) -> None:
    """Stream examples into an HF dataset at out_dir without holding the split in memory."""
    from datasets import Array2D, Dataset, Features, Sequence, Value

    feats = Features({"input_features": Array2D(shape=(n_mels, frames), dtype="float16"),
                      "labels": Sequence(Value("int64"))})
    cache = out_dir.with_name(f".{out_dir.name}.hf-cache")  # fresh per build: no cached generator reuse
    shutil.rmtree(cache, ignore_errors=True)
    try:
        ds = Dataset.from_generator(_RowGen(gen_rows), features=feats, cache_dir=str(cache),
                                    writer_batch_size=32)  # ~0.8 MB per row: keep write batches small
        ds.save_to_disk(str(out_dir))
    finally:
        shutil.rmtree(cache, ignore_errors=True)


def build_whisper(processor) -> Callable[[dict[str, list[dict]], Path], BuildResult]:
    def build(rows: dict[str, list[dict]], out: Path) -> BuildResult:
        load, extract, tokenize = _make_fns(processor, SAMPLE_RATE)
        fe = processor.feature_extractor
        counts: dict[str, int] = {}
        skipped: list[Skip] = []
        for split, items in rows.items():
            counts[split] = 0

            def examples(split=split, items=items) -> Iterator[dict]:
                for row in items:
                    ex = row_to_example(row, load=load, extract=extract, tokenize=tokenize)
                    if isinstance(ex, str):
                        skipped.append(Skip(sample_id=str(row.get("sample_id", "")), split=split, reason=ex))
                        continue
                    counts[split] += 1
                    yield ex

            _write_split(examples(), out / split, int(fe.feature_size), int(fe.nb_max_frames))
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
           "prepare_whisper"]
