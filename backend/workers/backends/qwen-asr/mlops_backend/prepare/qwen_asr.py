"""Qwen3-ASR converter (spec 17.3): 16 kHz mono WAV files plus one `<split>.jsonl` per split of
{"audio", "text", "duration", "sample_id"}.

`audio` is relative to the prepared dir: the cache is built in a temp dir and renamed into place, so absolute
paths would go stale. Text is `bn_normalize(mode="asr")`. Skipped: undecodable audio, clips over 30 s, empty
text, and texts whose target (transcript tokens + EOS) is over `max_label_tokens`.

Light module: numpy/soundfile/soxr load inside the audio helpers; the tokenizer is passed in as a callable.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from app.core.backend_kit.audio_refs import CHUNK, PROBE_THREADS, first_last_lines, progress_ticker
from app.core.prepare import BuildResult, PrepSpec, Skip, empty_split_error
from app.core.text import BN_NORMALIZE_VERSION, bn_normalize

from ..prompt import PREFIX, SAMPLE_RATE

FAMILY = "qwen-asr"
MAX_SECONDS = 30
MAX_LABEL_TOKENS = 448
AUDIO_DIR = "audio"


def tokenizer_hash(tokenizer) -> str:
    """sha256[:16] of the tokenizer vocabulary (added tokens included): a different tokenizer, a different cache."""
    blob = json.dumps(tokenizer.get_vocab(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def qwen_asr_spec(cfg, tok_hash: str, max_label_tokens: int = MAX_LABEL_TOKENS) -> PrepSpec:
    return PrepSpec(family=FAMILY, version=1, params={
        "base_model": cfg.base_model,
        "tokenizer_sha16": tok_hash,
        "prompt_prefix": PREFIX,
        "bn_normalize_version": BN_NORMALIZE_VERSION,
        "sample_rate": SAMPLE_RATE,
        "max_seconds": MAX_SECONDS,
        "max_label_tokens": int(max_label_tokens),
        "storage": "wav-16k-mono",
    })


def label_token_count(text: str, tokenize: Callable[[str], list[int]]) -> int:
    """Target tokens of one sample: the transcript plus EOS (the prompt and prefix are masked, not counted)."""
    return len(tokenize(text)) + 1


def text_skip(text: str, tokenize: Callable[[str], list[int]], max_label_tokens: int) -> str | None:
    if not text:
        return "empty text"
    n = label_token_count(text, tokenize)
    if n > max_label_tokens:
        return f"{n} label tokens > {max_label_tokens}"
    return None


def _default_loader(path):
    from app.core.backend_kit.audio import load_audio

    return load_audio(path, SAMPLE_RATE)


def _default_writer(path: Path, audio) -> None:
    from app.core.backend_kit.audio import write_wav

    write_wav(path, audio, SAMPLE_RATE)


def convert_clip(src: str, dest: Path, given_s: float | None, loader: Callable = _default_loader,
                 writer: Callable = _default_writer) -> float | str:
    """Decode `src` to 16 kHz mono and write it to `dest`. Returns the duration in seconds, or a skip reason."""
    if given_s and float(given_s) > MAX_SECONDS:  # the manifest already says too long: do not decode
        return f"duration {float(given_s):.1f} s > {MAX_SECONDS} s"
    try:
        audio = loader(src)
    except Exception as e:  # noqa: BLE001 - any decode failure skips the row (spec section 10)
        return f"undecodable audio: {type(e).__name__}"
    seconds = len(audio) / SAMPLE_RATE
    if seconds > MAX_SECONDS:
        return f"duration {seconds:.1f} s > {MAX_SECONDS} s"
    if seconds <= 0:
        return "empty audio"
    writer(dest, audio)
    return seconds


def build_qwen_asr(tokenize: Callable[[str], list[int]], rep=None, *, max_label_tokens: int = MAX_LABEL_TOKENS,
                   loader: Callable = _default_loader,
                   writer: Callable = _default_writer) -> Callable[[dict[str, list[dict]], Path], BuildResult]:
    def build(rows: dict[str, list[dict]], out: Path) -> BuildResult:
        from concurrent.futures import ThreadPoolExecutor

        out.mkdir(parents=True, exist_ok=True)
        tick = progress_ticker(rep, sum(len(v) for v in rows.values()))
        counts: dict[str, int] = {}
        skipped: list[Skip] = []
        with ThreadPoolExecutor(PROBE_THREADS) as pool:
            for split, items in rows.items():
                n = 0
                with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:  # streamed, row order kept
                    for lo in range(0, len(items), CHUNK):
                        chunk = items[lo:lo + CHUNK]
                        jobs = []
                        for i, row in enumerate(chunk, lo):
                            sid = str(row.get("sample_id", ""))
                            text = bn_normalize(row["text"], mode="asr")
                            reason = text_skip(text, tokenize, max_label_tokens)
                            if reason:
                                jobs.append((sid, text, None, reason))
                                continue
                            rel = f"{AUDIO_DIR}/{split}/{i:08d}.wav"
                            fut = pool.submit(convert_clip, str(row["audio_path"]), out / rel, row.get("duration_s"),
                                              loader, writer)
                            jobs.append((sid, text, rel, fut))
                        for sid, text, rel, res in jobs:
                            res = res.result() if rel is not None else res
                            if isinstance(res, str):
                                skipped.append(Skip(sample_id=sid, split=split, reason=res))
                                continue
                            f.write(json.dumps({"audio": rel, "text": text, "duration": round(float(res), 3),
                                                "sample_id": sid}, ensure_ascii=False) + "\n")
                            n += 1
                        tick(len(chunk))
                if n == 0:
                    raise empty_split_error(split, skipped)
                counts[split] = n
        return BuildResult(counts=counts, skipped=skipped, files=[f"{s}.jsonl" for s in rows] + [AUDIO_DIR])

    return build


def read_rows(prepared: Path, split: str) -> list[dict]:
    """Prepared rows of one split with `audio_path` resolved against the prepared dir."""
    prepared = Path(prepared)
    rows = []
    with (prepared / f"{split}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.append({"audio_path": str(prepared / r["audio"]), "text": r["text"],
                             "duration": float(r["duration"]), "sample_id": r.get("sample_id", "")})
    return rows


def check_prepared(prepared: Path, split: str = "train") -> None:
    """The first and last WAV of a split must still exist (reads two lines, not the file)."""
    prepared = Path(prepared)
    for r in first_last_lines(prepared / f"{split}.jsonl"):
        p = prepared / r["audio"]
        if not p.exists():
            raise RuntimeError(f"prepared cache {prepared} is missing {p}; delete the cache and re-run prepare")
