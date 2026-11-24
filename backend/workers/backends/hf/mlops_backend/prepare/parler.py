"""Parler-TTS converter. Prepared data stores DAC codes (not audio): per split a `datasets` directory with
{sample_id, prompt_input_ids, input_ids, codes (flat int16, frames-major), num_codebooks}. Codes are a small
fraction of the audio size, so this respects the disk budget; the build streams (Dataset.from_generator).

Light module: datasets, torch and numpy load only inside functions.
"""
from __future__ import annotations

import math
import shutil
from collections.abc import Callable
from pathlib import Path

from app.core.backend_kit.audio_refs import progress_ticker
from app.core.prepare import BuildResult, PrepareError, PrepSpec, Skip
from app.core.text import BN_NORMALIZE_VERSION, bn_normalize

SAMPLE_RATE = 44100
MAX_SECONDS = 30
BATCH = 16
IGNORE = -100
DEFAULT_TEMPLATE = "{speaker} speaks clearly in a studio-quality recording."


def describe(row: dict, template: str) -> str:
    """Voice description for a row: `{speaker}` and `{style}` filled (missing -> empty), spaces collapsed."""
    text = template.format(speaker=row.get("speaker") or "", style=row.get("style") or "")
    return " ".join(text.split())


def parler_spec(cfg, template: str) -> PrepSpec:
    return PrepSpec(family="parler", version=1, params={
        "base_model": cfg.base_model,
        "sample_rate": SAMPLE_RATE,
        "bn_normalize_version": BN_NORMALIZE_VERSION,
        "description_template": template,
        "max_seconds": MAX_SECONDS,
        "storage": "dac-codes",
    })


def row_to_example(row: dict, audio, template: str, tok_prompt: Callable, tok_desc: Callable,
                   sr: int = SAMPLE_RATE) -> dict | str:
    """One manifest row + its loaded audio (or the Exception load_audio raised) -> record without codes,
    or the skip reason (a str)."""
    if isinstance(audio, Exception):
        return f"undecodable audio: {type(audio).__name__}"
    seconds = len(audio) / sr
    if seconds > MAX_SECONDS:
        return f"duration {seconds:.1f} s > {MAX_SECONDS} s"
    text = bn_normalize(row["text"], mode="tts")
    if not text:
        return "empty text"
    return {"sample_id": str(row.get("sample_id", "")),
            "prompt_input_ids": [int(t) for t in tok_prompt(text)],
            "input_ids": [int(t) for t in tok_desc(describe(row, template))]}


def pad_code_labels(items: list[tuple[list[int], int]], eos_id: int) -> list[list[list[int]]]:
    """(flat frames-major codes, num_codebooks) per row -> (rows, T+1, K): an EOS frame appended, shorter rows
    padded with -100 (the loss ignores them; the model shifts in its own BOS)."""
    seqs = []
    for flat, k in items:
        frames = [list(flat[i:i + k]) for i in range(0, len(flat), k)]
        frames.append([eos_id] * k)
        seqs.append((frames, k))
    width = max((len(f) for f, _ in seqs), default=0)
    return [f + [[IGNORE] * k] * (width - len(f)) for f, k in seqs]


def _hop(audio_encoder) -> int:
    ratios = getattr(audio_encoder.config, "upsampling_ratios", None)
    return int(math.prod(ratios)) if ratios else 512


def make_encoder(feature_extractor, audio_encoder) -> Callable[[list], list]:
    """audios (float32 arrays at 44.1 kHz) -> per-row DAC codes of shape (codebooks, frames), on GPU when
    available. Rows are zero-padded to the batch's longest and trimmed to their own frame count."""
    import numpy as np
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_encoder.to(device).eval()
    hop = _hop(audio_encoder)

    def encode(audios: list) -> list:
        b = feature_extractor(list(audios), sampling_rate=SAMPLE_RATE, padding="longest", return_tensors="pt")
        with torch.no_grad():
            out = audio_encoder.encode(b["input_values"].to(device), b["padding_mask"].to(device))
        codes = out.audio_codes
        codes = codes[0] if codes.ndim == 4 else codes  # (1, bsz, K, T) -> (bsz, K, T)
        codes = codes.cpu().numpy()
        return [codes[i, :, :math.ceil(len(a) / hop)].astype(np.int16) for i, a in enumerate(audios)]

    return encode


def _load(loader: Callable, path):
    try:
        return loader(path)
    except Exception as e:  # noqa: BLE001 - reported as an undecodable skip
        return e


def _default_loader(path):
    from app.core.backend_kit.audio import load_audio

    return load_audio(path, SAMPLE_RATE)


def build_parler(feature_extractor, audio_encoder, prompt_tokenizer, description_tokenizer, template: str,
                 rep=None, *, encode: Callable | None = None,
                 loader: Callable = _default_loader) -> Callable[[dict[str, list[dict]], Path], BuildResult]:
    def tok_prompt(t):
        return prompt_tokenizer(t).input_ids

    def tok_desc(t):
        return description_tokenizer(t).input_ids

    def build(rows: dict[str, list[dict]], out: Path) -> BuildResult:
        from concurrent.futures import ThreadPoolExecutor

        from datasets import Dataset, Features, Sequence, Value

        enc = encode or make_encoder(feature_extractor, audio_encoder)
        out.mkdir(parents=True, exist_ok=True)
        tick = progress_ticker(rep, sum(len(v) for v in rows.values()))
        feats = Features({"sample_id": Value("string"), "prompt_input_ids": Sequence(Value("int32")),
                          "input_ids": Sequence(Value("int32")), "codes": Sequence(Value("int16")),
                          "num_codebooks": Value("int32")})
        counts: dict[str, int] = {}
        skipped: list[Skip] = []
        with ThreadPoolExecutor(8) as pool:
            for split, items in rows.items():
                def gen(items=items, split=split):
                    for lo in range(0, len(items), BATCH):
                        chunk = items[lo:lo + BATCH]
                        audios = list(pool.map(lambda r: _load(loader, r["audio_path"]), chunk))
                        good = []
                        for row, audio in zip(chunk, audios):
                            ex = row_to_example(row, audio, template, tok_prompt, tok_desc)
                            if isinstance(ex, str):
                                skipped.append(Skip(sample_id=str(row.get("sample_id", "")), split=split, reason=ex))
                            else:
                                good.append((ex, audio))
                        if good:
                            codes = enc([a for _, a in good])
                            for (ex, _), c in zip(good, codes):
                                yield {**ex, "codes": c.T.reshape(-1), "num_codebooks": int(c.shape[0])}
                        tick(len(chunk))

                tmp = out / f".gen-{split}"  # fresh cache dir: never a stale fingerprint hit that skips gen()
                ds = Dataset.from_generator(gen, features=feats, cache_dir=str(tmp))
                if len(ds) == 0:
                    raise PrepareError(f"split {split} is empty after prepare ({len(skipped)} rows skipped)")
                ds.save_to_disk(str(out / split))
                counts[split] = len(ds)
                del ds
                shutil.rmtree(tmp, ignore_errors=True)
        return BuildResult(counts=counts, skipped=skipped, files=list(rows))

    return build
