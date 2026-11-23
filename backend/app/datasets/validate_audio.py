from __future__ import annotations

import csv
import hashlib
import re
import unicodedata
from pathlib import Path

import numpy as np
import pyloudnorm
import soundfile as sf

from ..core.splits import normalize_split
from .report import Issue, Report

_REQUIRED = {"tts": ["file", "text", "speaker"], "asr": ["file", "text"]}
_AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3"}
_RESERVED = {"sample_id", "duration_s", "sample_rate"}
_MIN_S, _MAX_S = 0.5, 30.0


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_META_SPLIT_RE = re.compile(r"^metadata_(.+)\.csv$", re.IGNORECASE)
SPLIT_ERR = "use train, valid or test"


def metadata_files(root: Path) -> tuple[Path | None, dict[str, Path], list[Issue]]:
    """metadata.csv (if any), the recognised metadata_<split>.csv files keyed by split, and layout problems."""
    plain = root / "metadata.csv"
    per_split: dict[str, Path] = {}
    problems: list[Issue] = []
    for p in sorted(root.iterdir()):
        m = _META_SPLIT_RE.match(p.name)
        if not (m and p.is_file()):
            continue
        try:
            sp = normalize_split(m.group(1))
        except ValueError:
            sp = None
        if sp is None:
            problems.append(Issue("unknown_split_file", f"split file '{p.name}': {SPLIT_ERR}"))
        elif sp in per_split:
            problems.append(Issue("duplicate_split_file", f"duplicate split files for '{sp}': "
                                  f"{per_split[sp].name} and {p.name}"))
        else:
            per_split[sp] = p
    return (plain if plain.is_file() else None), per_split, problems


def _read_records(path: Path) -> tuple[list[str], list[tuple[int, dict]]]:
    # utf-8-sig strips the BOM Excel writes; newline="" lets csv handle CRLF
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = [(f or "").strip() for f in (reader.fieldnames or [])]
        reader.fieldnames = fieldnames
        records = []
        for r in reader:
            if any(isinstance(v, str) and v.strip() for v in r.values()):
                records.append((reader.line_num, r))
    return fieldnames, records


def validate_audio(root: Path, task: str) -> tuple[Report, list[dict]]:
    report = Report()
    report.stats.update(n_samples=0, total_duration_s=0.0)
    plain, per_split, problems = metadata_files(root)
    if problems:
        report.errors.extend(problems)
        return report, []
    if plain is not None and per_split:
        report.errors.append(Issue(
            "mixed_metadata", "both metadata.csv and metadata_<split>.csv found: use one layout"))
        return report, []
    if plain is None and not per_split:
        report.errors.append(Issue("missing_metadata", "metadata.csv not found at the dataset root"))
        return report, []
    sources: list[tuple[Path, str | None]] = [(plain, None)] if plain else [
        (per_split[sp], sp) for sp in ("train", "val", "test") if sp in per_split]
    records: list[tuple[int, dict, str | None, str]] = []  # line, record, file split, file name
    for path, file_split in sources:
        try:
            fieldnames, recs = _read_records(path)
        except UnicodeDecodeError:
            report.errors.append(Issue("missing_metadata", f"{path.name} is not valid UTF-8"))
            return report, []
        missing = [c for c in _REQUIRED[task] if c not in fieldnames]
        for c in missing:
            report.errors.append(Issue("missing_column", f"{path.name} has no '{c}' column"))
        if missing:
            return report, []
        records += [(ln, r, file_split, path.name) for ln, r in recs]

    # Index audio on disk by NFC-normalized relative path (Windows and macOS normalize differently).
    on_disk: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in _AUDIO_EXT:
            on_disk[_nfc(p.relative_to(root).as_posix())] = p
    referenced: set[str] = set()

    rows: list[dict] = []
    rates: set[int] = set()
    for line_no, rec, file_split, meta_name in records:
        rel = _nfc((rec.get("file") or "").strip().replace("\\", "/"))
        text = (rec.get("text") or "").strip()
        raw_split = (rec.get("split") or "").strip()
        if file_split is not None:
            if raw_split:
                report.errors.append(Issue(
                    "split_in_split_file", f"{meta_name} line {line_no}: row has a split value but its file already sets the split"))
                referenced.add(rel)
                continue
            split = file_split
        else:
            try:
                split = normalize_split(raw_split)
            except ValueError:
                report.errors.append(Issue(
                    "unknown_split", f"split value '{raw_split}' on line {line_no}: {SPLIT_ERR}"))
                referenced.add(rel)
                continue
        path = on_disk.get(rel)
        if path is None:
            report.errors.append(
                Issue("missing_audio", f"'{rel}' is listed in {meta_name} but not found in the upload")
            )
            continue
        if rel in referenced:
            report.errors.append(Issue("duplicate_file", f"'{rel}' is listed more than once in the metadata"))
            continue
        referenced.add(rel)
        if not text:
            report.errors.append(Issue("missing_transcript", f"'{rel}' has an empty transcript"))
            continue
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception as e:  # noqa: BLE001 - any decode failure is a data error
            report.errors.append(Issue("undecodable", f"'{rel}' cannot be decoded: {e}"))
            continue
        mono = data.mean(axis=1)
        duration = len(mono) / sr
        sample_id = hashlib.sha256((text + "\n" + _sha256_file(path)).encode("utf-8")).hexdigest()[:16]
        row = {"sample_id": sample_id, "file": rel, "text": text,
               "duration_s": round(duration, 6), "sample_rate": int(sr), "split": split}
        for k, v in rec.items():
            if k and k not in row and k not in _RESERVED and isinstance(v, str):
                row[k] = v
        rows.append(row)
        rates.add(int(sr))
        report.stats["total_duration_s"] += duration
        _audio_warnings(report, sample_id, rel, mono, int(sr), duration)

    for rel in sorted(set(on_disk) - referenced):
        report.errors.append(Issue("missing_transcript", f"'{rel}' has no row in the metadata"))
    if len(rates) > 1:
        report.warnings.append(Issue("mixed_sample_rates", f"sample rates differ: {sorted(rates)}"))
    report.stats["n_samples"] = len(rows)
    report.stats["total_duration_s"] = round(report.stats["total_duration_s"], 3)
    report.stats["sample_rates"] = sorted(rates)
    return report, rows


def _audio_warnings(report: Report, sid: str, rel: str, x: np.ndarray, sr: int, duration: float) -> None:
    def warn(code: str, msg: str) -> None:
        report.warnings.append(Issue(code, f"{rel}: {msg}", sid))

    if not _MIN_S <= duration <= _MAX_S:
        warn("duration_out_of_range", f"{duration:.2f}s is outside {_MIN_S}-{_MAX_S}s")
    if len(x) and np.mean(np.abs(x) >= 0.999) > 0.001:
        warn("clipping", "audio is clipped")
    frame = max(1, int(sr * 0.02))
    n = len(x) // frame
    if n:
        frames = x[: n * frame].reshape(n, frame).astype(np.float64)
        rms = np.sqrt(np.mean(frames**2, axis=1))
        silent = float(np.mean(rms < 10 ** (-50 / 20)))
        if silent > 0.5:
            warn("high_silence", f"{silent:.0%} of 20 ms frames are below -50 dBFS")
    if duration >= 0.4:
        with np.errstate(all="ignore"):
            lufs = pyloudnorm.Meter(sr).integrated_loudness(x.astype(np.float64))
        if not -30 <= lufs <= -14:
            warn("loudness_out_of_range", f"{lufs:.1f} LUFS is outside -30..-14")
