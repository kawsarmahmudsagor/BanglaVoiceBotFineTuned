"""CTC (wav2vec2 / MMS) converter. Prepared data stores references only (no audio): `<split>.jsonl` rows of
{"audio_path", "duration_s", "text"} plus `vocab.json`. Audio is decoded and resampled lazily by the collator.

Light module: numpy/soundfile load only inside the shared probe, transformers only inside mms_vocab.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal

from app.core.backend_kit.audio_refs import (  # noqa: F401
    PROGRESS_EVERY, SAMPLE_RATE, first_last_lines, probe_duration, probed_rows, progress_ticker,
)
from app.core.prepare import BuildResult, PrepareError, PrepSpec, Skip
from app.core.text import BN_NORMALIZE_VERSION, bn_normalize

MAX_SECONDS = 30
IGNORE = -100
UNK, PAD, DELIM = "[UNK]", "[PAD]", "|"


def build_ctc_vocab(texts: Iterable[str]) -> dict[str, int]:
    """Characters of the asr-normalized texts (one pass), sorted, space -> `|`, then [UNK] and [PAD] last."""
    chars: set[str] = set()
    for t in texts:
        chars.update(bn_normalize(t, mode="asr").replace(" ", DELIM))
    ordered = sorted(chars) + [UNK, PAD]
    return {c: i for i, c in enumerate(ordered)}


def vocab_hash(vocab: dict) -> str:
    blob = json.dumps(vocab, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def ctc_spec(cfg, vocab: dict) -> PrepSpec:
    """The vocab hash is part of the key, so a warm start from another run's vocab gets its own cache."""
    return PrepSpec(family="ctc", version=1, params={
        "base_model": cfg.base_model,
        "sample_rate": SAMPLE_RATE,
        "bn_normalize_version": BN_NORMALIZE_VERSION,
        "max_seconds": MAX_SECONDS,
        "vocab_sha16": vocab_hash(vocab),
        "storage": "audio-ref",
    })


def special_ids(vocab: dict[str, int]) -> tuple[int, int]:
    """(unk id, pad id); MMS vocabularies spell them `<unk>`/`<pad>`."""
    unk = vocab.get(UNK, vocab.get("<unk>"))
    pad = vocab.get(PAD, vocab.get("<pad>"))
    if unk is None or pad is None:
        raise ValueError("vocab has no unknown/pad token")
    return int(unk), int(pad)


def ctc_labels(text: str, vocab: dict[str, int]) -> list[int]:
    unk, _ = special_ids(vocab)
    return [vocab.get(c, unk) for c in text.replace(" ", DELIM)]


def pad_ctc_labels(label_lists: list[list[int]]) -> list[list[int]]:
    width = max((len(r) for r in label_lists), default=0)
    return [r + [IGNORE] * (width - len(r)) for r in label_lists]


def ctc_greedy_decode(ids: list[int], inv_vocab: dict[int, str], pad_id: int, unk_id: int | None = None) -> str:
    """Collapse repeats, drop blanks (pad) and unknowns, `|` -> space."""
    out: list[str] = []
    prev = None
    for i in ids:
        if i != prev and i != pad_id and i != unk_id:
            out.append(inv_vocab.get(int(i), ""))
        prev = i
    return "".join(out).replace(DELIM, " ").strip()


def _row_result(row: dict, res, text: str, vocab: dict, mode: str) -> dict | str:
    if isinstance(res, Exception):
        return f"undecodable audio: {type(res).__name__}"
    given = row.get("duration_s")
    seconds = float(given) if given else float(res)
    if seconds > MAX_SECONDS:
        return f"duration {seconds:.1f} s > {MAX_SECONDS} s"
    if not text:
        return "empty text"
    if mode == "fixed":
        missing = sorted(set(text.replace(" ", DELIM)) - vocab.keys())
        if missing:
            return f"characters not in vocab: {''.join(missing)}"
    return {"audio_path": str(row["audio_path"]), "duration_s": seconds, "text": text}


def build_ctc(vocab: dict[str, int], mode: Literal["new", "fixed"], rep=None,
              probe: Callable = probe_duration) -> Callable[[dict[str, list[dict]], Path], BuildResult]:
    """`new`: the vocab was built from this data (unseen characters are never skipped). `fixed`: the vocab is
    imposed (warm start / MMS), rows with characters outside it are skipped."""
    def build(rows: dict[str, list[dict]], out: Path) -> BuildResult:
        out.mkdir(parents=True, exist_ok=True)
        (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=1), encoding="utf-8")
        tick = progress_ticker(rep, sum(len(v) for v in rows.values()))
        counts: dict[str, int] = {}
        skipped: list[Skip] = []
        for split, items in rows.items():
            n = 0
            with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:  # streamed row by row
                for row, res in probed_rows(items, probe, tick):
                    text = bn_normalize(row["text"], mode="asr")
                    ex = _row_result(row, res, text, vocab, mode)
                    if isinstance(ex, str):
                        skipped.append(Skip(sample_id=str(row.get("sample_id", "")), split=split, reason=ex))
                        continue
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    n += 1
            if n == 0:
                raise PrepareError(f"split {split} is empty after prepare ({len(skipped)} rows skipped)")
            counts[split] = n
        return BuildResult(counts=counts, skipped=skipped, files=[f"{s}.jsonl" for s in rows] + ["vocab.json"])

    return build


def mms_vocab(processor) -> dict[str, int]:
    """Vocabulary of MMS's `ben` target language (heavy: needs the real processor)."""
    processor.tokenizer.set_target_lang("ben")
    return {k: int(v) for k, v in processor.tokenizer.get_vocab().items()}


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
