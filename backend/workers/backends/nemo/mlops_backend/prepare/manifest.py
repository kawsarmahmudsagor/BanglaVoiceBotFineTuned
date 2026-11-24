"""NeMo manifest converter. The manifest REFERENCES the source audio (no copy, disk budget): `<split>.jsonl`
lines {"audio_filepath", "duration", "text"}. NeMo resamples at load (train_ds.sample_rate=16000). Only a source
soundfile cannot open but a fallback decoder can is transcoded to 16 kHz mono FLAC in `<out>/audio/`.

Light module: numpy/soundfile load only inside functions.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from app.core.backend_kit.audio_refs import SAMPLE_RATE, probed_rows, progress_ticker
from app.core.prepare import BuildResult, PrepareError, PrepSpec, Skip
from app.core.text import BN_NORMALIZE_VERSION, bn_normalize

MAX_SECONDS = 30


class Decoded:
    """Probe result for a file soundfile cannot open but a fallback decoder can: the 16 kHz mono samples."""

    def __init__(self, audio):
        self.audio = audio


def make_probe(loader: Callable | None = None) -> Callable:
    """probe(path) -> seconds (soundfile header) | Decoded (fallback decode) | raises."""
    def probe(path):
        import soundfile as sf

        try:
            return float(sf.info(str(path)).duration)
        except Exception:  # noqa: BLE001 - not readable by soundfile: try the fallback decoder
            load = loader
            if load is None:
                from app.core.backend_kit.audio import load_audio as load
            return Decoded(load(path, SAMPLE_RATE))

    return probe


def tokenizer_hash(vocab) -> str:
    blob = json.dumps(vocab, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def manifest_spec(cfg, tok_sha16: str) -> PrepSpec:
    return PrepSpec(family="nemo-manifest", version=1, params={
        "base_model": cfg.base_model,
        "sample_rate": SAMPLE_RATE,
        "bn_normalize_version": BN_NORMALIZE_VERSION,
        "max_seconds": MAX_SECONDS,
        "tokenizer_sha16": tok_sha16,
        "storage": "audio-ref",
    })


def _unk_chars(text: str, encode: Callable, unk_id: int) -> str:
    seen: dict[str, None] = {}
    for c in text:
        if c not in seen:
            seen[c] = None
    return "".join(c for c in seen if unk_id in encode(c))


def build_manifest(encode: Callable[[str], list[int]] | None, unk_id: int | None, rep=None,
                   probe: Callable | None = None,
                   loader: Callable | None = None) -> Callable[[dict[str, list[dict]], Path], BuildResult]:
    probe = probe or make_probe(loader)

    def build(rows: dict[str, list[dict]], out: Path) -> BuildResult:
        out.mkdir(parents=True, exist_ok=True)
        tick = progress_ticker(rep, sum(len(v) for v in rows.values()))
        counts: dict[str, int] = {}
        skipped: list[Skip] = []
        for split, items in rows.items():
            n = 0
            with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
                for row, res in probed_rows(items, probe, tick):
                    sid = str(row.get("sample_id", ""))
                    reason, rec = _row(row, res, sid, out, encode, unk_id)
                    if reason:
                        skipped.append(Skip(sample_id=sid, split=split, reason=reason))
                        continue
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
            if n == 0:
                raise PrepareError(f"split {split} is empty after prepare ({len(skipped)} rows skipped)")
            counts[split] = n
        return BuildResult(counts=counts, skipped=skipped, files=[f"{s}.jsonl" for s in rows])

    return build


def _row(row, res, sid, out, encode, unk_id) -> tuple[str | None, dict | None]:
    if isinstance(res, Exception):
        return f"undecodable audio: {type(res).__name__}", None
    if isinstance(res, Decoded):
        seconds = len(res.audio) / SAMPLE_RATE
    else:
        seconds = float(res or row.get("duration_s") or 0)
    if seconds > MAX_SECONDS:
        return f"duration {seconds:.1f} s > {MAX_SECONDS} s", None
    text = bn_normalize(row["text"], mode="asr")
    if not text:
        return "empty text", None
    if encode is not None and unk_id is not None and unk_id in encode(text):
        return f"tokenizer maps characters to <unk>: {_unk_chars(text, encode, unk_id)}", None
    path = str(Path(row["audio_path"]).resolve())
    if isinstance(res, Decoded):
        from app.core.backend_kit.audio import write_flac

        flac = out / "audio" / f"{sid}.flac"
        write_flac(flac, res.audio, SAMPLE_RATE)
        path = str(flac.resolve())
    return None, {"audio_filepath": path, "duration": round(seconds, 3), "text": text}
