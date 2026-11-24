"""F5-TTS converter. Per split `<split>/raw.arrow` (Arrow IPC stream: audio_path, text, duration) and
`<split>/duration.json`; one `vocab.txt` (base vocab + new characters); `refs/<speaker>.wav|txt` reference clips.

raw.arrow REFERENCES the source audio (no copy, disk budget): f5-tts' CustomDataset loads `audio_path` with
torchaudio and resamples to its 24 kHz target at load. Only a source soundfile cannot open but a fallback decoder
can is transcoded to 24 kHz mono FLAC in `<out>/wavs/`. Light module: numpy/soundfile/pyarrow load inside functions.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath

from app.core.backend_kit.audio_refs import probed_rows, progress_ticker
from app.core.prepare import BuildResult, PrepareError, PrepSpec, Skip, empty_split_error
from app.core.text import BN_NORMALIZE_VERSION, bn_normalize
from app.core.tts_refs import DEFAULT_SPEAKER, pick_reference

SAMPLE_RATE = 24000
MAX_SECONDS = 30


class Decoded:
    """Probe result for a file soundfile cannot open but a fallback decoder can: the 24 kHz mono samples."""

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


def extend_vocab(base: list[str], texts: Iterable[str]) -> list[str]:
    """`base` unchanged, then the characters of `texts` not in it, in first-seen order."""
    seen = set(base)
    out = list(base)
    for text in texts:
        for c in text:
            if c not in seen and c != "\n":
                seen.add(c)
                out.append(c)
    return out


def read_vocab(path: Path) -> list[str]:
    """One character per line (a line holding a space is the space character)."""
    lines = Path(path).read_bytes().decode("utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def write_vocab(path: Path, vocab: list[str]) -> None:
    Path(path).write_bytes(("\n".join(vocab) + "\n").encode("utf-8"))


def vocab_hash(vocab: list[str]) -> str:
    blob = json.dumps(vocab, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def pick_vocab_file(files: list[str]) -> str:
    """The vocab file of a hub repo: `vocab*.txt`, preferring the shallowest path, then `vocab.txt` itself."""
    cands = [f for f in files if re.fullmatch(r"vocab.*\.txt", PurePosixPath(f).name)]
    if not cands:
        raise PrepareError("no vocab.txt found in the model repo")
    return min(cands, key=lambda f: (f.count("/"), PurePosixPath(f).name != "vocab.txt", f))


def f5_spec(cfg, base_vocab_sha16: str) -> PrepSpec:
    return PrepSpec(family="f5-arrow", version=1, params={
        "base_model": cfg.base_model,
        "sample_rate": SAMPLE_RATE,
        "bn_normalize_version": BN_NORMALIZE_VERSION,
        "max_seconds": MAX_SECONDS,
        "base_vocab_sha16": base_vocab_sha16,
        "storage": "audio-ref",
    })


def _safe(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]', "_", name) or DEFAULT_SPEAKER


def _row(row, res, sid, out) -> tuple[str | None, dict | None]:
    if isinstance(res, Exception):
        return f"undecodable audio: {type(res).__name__}", None
    seconds = len(res.audio) / SAMPLE_RATE if isinstance(res, Decoded) else float(res or row.get("duration_s") or 0)
    if seconds > MAX_SECONDS:
        return f"duration {seconds:.1f} s > {MAX_SECONDS} s", None
    text = bn_normalize(row["text"], mode="tts")
    if not text:
        return "empty text", None
    path = str(Path(row["audio_path"]).resolve())
    if isinstance(res, Decoded):
        from app.core.backend_kit.audio import write_flac

        flac = out / "wavs" / f"{sid}.flac"
        write_flac(flac, res.audio, SAMPLE_RATE)
        path = str(flac.resolve())
    return None, {"sample_id": sid, "speaker": str(row.get("speaker") or DEFAULT_SPEAKER), "audio_path": path,
                  "text": text, "duration": round(seconds, 3), "duration_s": round(seconds, 3)}


def _write_split(out: Path, split: str, recs: list[dict]) -> None:
    import pyarrow as pa

    d = out / split
    d.mkdir(parents=True, exist_ok=True)
    table = pa.table({"audio_path": [r["audio_path"] for r in recs], "text": [r["text"] for r in recs],
                      "duration": [r["duration"] for r in recs]})
    with pa.OSFile(str(d / "raw.arrow"), "wb") as f, pa.ipc.new_stream(f, table.schema) as w:
        w.write_table(table)
    (d / "duration.json").write_text(json.dumps({"duration": [r["duration"] for r in recs]}), encoding="utf-8")


def _write_refs(out: Path, recs: list[dict], load: Callable) -> None:
    from app.core.backend_kit.audio import write_wav

    for spk, r in pick_reference(recs).items():
        name = _safe(spk)
        write_wav(out / "refs" / f"{name}.wav", load(r["audio_path"], SAMPLE_RATE), SAMPLE_RATE)
        (out / "refs" / f"{name}.txt").write_text(r["text"], encoding="utf-8")


def build_f5(base_vocab: list[str], rep=None, probe: Callable | None = None,
             loader: Callable | None = None) -> Callable[[dict[str, list[dict]], Path], BuildResult]:
    probe = probe or make_probe(loader)

    def build(rows: dict[str, list[dict]], out: Path) -> BuildResult:
        out.mkdir(parents=True, exist_ok=True)
        load = loader
        if load is None:
            from app.core.backend_kit.audio import load_audio as load
        tick = progress_ticker(rep, sum(len(v) for v in rows.values()))
        counts: dict[str, int] = {}
        skipped: list[Skip] = []
        kept: dict[str, list[dict]] = {}
        for split, items in rows.items():
            recs: list[dict] = []
            for row, res in probed_rows(items, probe, tick):
                sid = str(row.get("sample_id", ""))
                reason, rec = _row(row, res, sid, out)
                if reason:
                    skipped.append(Skip(sample_id=sid, split=split, reason=reason))
                else:
                    recs.append(rec)
            if not recs:
                raise empty_split_error(split, skipped)
            kept[split] = recs
            counts[split] = len(recs)
        allrecs = [r for recs in kept.values() for r in recs]
        for split, recs in kept.items():
            _write_split(out, split, recs)
        write_vocab(out / "vocab.txt", extend_vocab(base_vocab, (r["text"] for r in allrecs)))
        _write_refs(out, allrecs, load)
        files = [f"{s}/{n}" for s in kept for n in ("raw.arrow", "duration.json")] + ["vocab.txt"]
        return BuildResult(counts=counts, skipped=skipped, files=files)

    return build
