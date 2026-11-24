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


def _hop(audio_encoder) -> float:
    """Samples per code frame: sampling_rate / frame_rate when the config has them, else the product of the
    upsampling ratios, else 512."""
    conf = audio_encoder.config
    rate, fps = getattr(conf, "sampling_rate", None), getattr(conf, "frame_rate", None)
    if rate and fps:
        return float(rate) / float(fps)
    ratios = getattr(conf, "upsampling_ratios", None)
    return float(math.prod(ratios)) if ratios else 512.0


def trim_codes(codes, lengths: list[int], hop: float) -> list:
    """codes (bsz, K, T) from a zero-padded batch -> per-row (K, ceil(len/hop)) int16 arrays."""
    import numpy as np

    return [np.asarray(codes[i, :, :math.ceil(n / hop)]).astype(np.int16) for i, n in enumerate(lengths)]


def encoder_inputs(batch, device) -> tuple:
    """(input_values, padding_mask or None) on `device`. The DAC feature extractor of transformers 4.46 returns no
    padding_mask; parler's DAC encode() treats it as optional (all ones), and trim_codes cuts each row to its length."""
    mask = batch.get("padding_mask")
    return batch["input_values"].to(device), (None if mask is None else mask.to(device))


def make_encoder(feature_extractor, audio_encoder) -> Callable[[list], list]:
    """audios (float32 arrays at 44.1 kHz) -> per-row DAC codes of shape (codebooks, frames), on GPU when
    available. Rows are zero-padded to the batch's longest and trimmed to their own frame count."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_encoder.to(device).eval()
    hop = _hop(audio_encoder)

    def encode(audios: list) -> list:
        b = feature_extractor(list(audios), sampling_rate=SAMPLE_RATE, padding="longest", return_tensors="pt")
        with torch.no_grad():
            out = audio_encoder.encode(*encoder_inputs(b, device))
        codes = out.audio_codes
        codes = codes[0] if codes.ndim == 4 else codes  # (1, bsz, K, T) -> (bsz, K, T)
        codes = codes.cpu().numpy()
        return trim_codes(codes, [len(a) for a in audios], hop)

    return encode


def _load(loader: Callable, path):
    try:
        return loader(path)
    except Exception as e:  # noqa: BLE001 - reported as an undecodable skip
        return e


def _default_loader(path):
    from app.core.backend_kit.audio import load_audio

    return load_audio(path, SAMPLE_RATE)


SHARD_ROWS = 512
MAX_TOKENS = 256


def iter_records(items: list[dict], split: str, skipped: list, enc: Callable, loader: Callable, template: str,
                 tok_prompt: Callable, tok_desc: Callable, tick: Callable[[int], None] = lambda n: None):
    """Stream prepared records (codes flat frames-major int16) for one split; skip reasons go to `skipped`.
    Light: no datasets/torch. Audio is held for one batch only."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(8) as pool:
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
                for (ex, _), c in zip(good, enc([a for _, a in good])):
                    yield {**ex, "codes": c.T.reshape(-1), "num_codebooks": int(c.shape[0])}
            tick(len(chunk))


def write_split(records, out: Path, shard_rows: int = SHARD_ROWS) -> int:
    """Stream records into small Dataset shards (from_dict, no generator pickling) and join them into one
    `datasets` directory at `out`. Returns the row count."""
    from datasets import Dataset, Features, Sequence, Value, concatenate_datasets, load_from_disk

    feats = Features({"sample_id": Value("string"), "prompt_input_ids": Sequence(Value("int32")),
                      "input_ids": Sequence(Value("int32")), "codes": Sequence(Value("int16")),
                      "num_codebooks": Value("int32")})
    shards_dir = out.with_name(out.name + ".shards")
    shutil.rmtree(shards_dir, ignore_errors=True)
    shards: list[Path] = []
    buf: list[dict] = []

    def flush():
        if buf:
            d = shards_dir / f"{len(shards):05d}"
            Dataset.from_dict({k: [r[k] for r in buf] for k in feats}, features=feats).save_to_disk(str(d))
            shards.append(d)
            buf.clear()

    for r in records:
        buf.append(r)
        if len(buf) >= shard_rows:
            flush()
    flush()
    n = 0
    if shards:
        ds = concatenate_datasets([load_from_disk(str(d)) for d in shards])
        n = len(ds)
        ds.save_to_disk(str(out))
        del ds
    shutil.rmtree(shards_dir, ignore_errors=True)
    return n


def build_parler(feature_extractor, audio_encoder, prompt_tokenizer, description_tokenizer, template: str,
                 rep=None, *, encode: Callable | None = None,
                 loader: Callable = _default_loader) -> Callable[[dict[str, list[dict]], Path], BuildResult]:
    def tok_prompt(t):
        return prompt_tokenizer(t, truncation=True, max_length=MAX_TOKENS).input_ids

    def tok_desc(t):
        return description_tokenizer(t, truncation=True, max_length=MAX_TOKENS).input_ids

    def build(rows: dict[str, list[dict]], out: Path) -> BuildResult:
        enc = encode or make_encoder(feature_extractor, audio_encoder)
        out.mkdir(parents=True, exist_ok=True)
        tick = progress_ticker(rep, sum(len(v) for v in rows.values()))
        counts: dict[str, int] = {}
        skipped: list[Skip] = []
        for split, items in rows.items():
            recs = iter_records(items, split, skipped, enc, loader, template, tok_prompt, tok_desc, tick)
            n = write_split(recs, out / split)
            if n == 0:
                raise PrepareError(f"split {split} is empty after prepare ({len(skipped)} rows skipped)")
            counts[split] = n
        return BuildResult(counts=counts, skipped=skipped, files=list(rows))

    return build
