from __future__ import annotations

import hashlib
import json
import shutil
import unicodedata
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.parquet as pq

from .report import Issue, Report
from .validate_audio import assign_split, validate_audio
from .validate_llm import validate_llm

_TASK_EXT = {"llm-sft": ".jsonl", "asr": ".zip", "tts": ".zip"}


@dataclass
class IngestResult:
    report: Report
    version_hash: str | None
    version_dir: Path | None
    n_samples: int


def _member_name(info: zipfile.ZipInfo) -> str:
    name = info.filename
    if not info.flag_bits & 0x800:  # no UTF-8 flag: zipfile decoded the name as cp437
        try:
            name = name.encode("cp437").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return name.replace("\\", "/")


def _unsafe(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return name.startswith("/") or ".." in parts or bool(parts and len(parts[0]) >= 2 and parts[0][1] == ":")


def _extract(upload: Path, dest: Path, report: Report) -> Path | None:
    """Extract safely; return the dataset root, or None after recording errors."""
    try:
        zf = zipfile.ZipFile(upload)
    except zipfile.BadZipFile:
        report.errors.append(Issue("unsupported_format", "upload is not a valid zip archive"))
        return None
    with zf:
        members = [(i, _member_name(i)) for i in zf.infolist()]
        bad = [n for _, n in members if _unsafe(n)]
        for n in bad:
            report.errors.append(Issue("unsafe_path", f"archive member '{n}' escapes the dataset directory"))
        if bad:
            return None
        for info, name in members:
            if name.endswith("/") or info.is_dir():
                continue
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
    if (dest / "metadata.csv").is_file():
        return dest
    tops = [p for p in dest.iterdir() if p.name != "__MACOSX"]
    if len(tops) == 1 and tops[0].is_dir() and (tops[0] / "metadata.csv").is_file():
        return tops[0]
    return dest


def _version_hash(rows: list[dict]) -> str:
    lines = [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["sample_id"], lines[i]))
    h = hashlib.sha256()
    for i in order:
        h.update(lines[i].encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _locate(root: Path, rel: str) -> Path:
    direct = root / rel
    if direct.is_file():
        return direct
    for p in root.rglob("*"):  # name stored in a different Unicode normalization form
        if p.is_file() and unicodedata.normalize("NFC", p.relative_to(root).as_posix()) == rel:
            return p
    raise FileNotFoundError(rel)


def ingest(upload_file: Path, task: str, dest_root: Path) -> IngestResult:
    upload_file = Path(upload_file)
    dest_root = Path(dest_root)
    want = _TASK_EXT.get(task)
    if want is None or upload_file.suffix.lower() != want:
        msg = f"unknown task '{task}'" if want is None else f"task '{task}' expects a {want} upload"
        return IngestResult(Report(errors=[Issue("unsupported_format", msg)]), None, None, 0)

    staging = dest_root / ".staging" / uuid.uuid4().hex
    staging.mkdir(parents=True)
    try:
        root: Path | None = None
        if task == "llm-sft":
            report, rows = validate_llm(upload_file)
        else:
            extracted = staging / "extracted"
            extracted.mkdir()
            report = Report()
            rows = []
            root = _extract(upload_file, extracted, report)
            if root is not None:
                report, rows = validate_audio(root, task)
        if not report.ok:
            return IngestResult(report, None, None, 0)

        splits = {"train": 0, "val": 0, "test": 0}
        for r in rows:
            r["split"] = assign_split(r["sample_id"])
            splits[r["split"]] += 1
        report.stats["splits"] = splits
        report.stats["n_samples"] = len(rows)

        vhash = _version_hash(rows)[:16]
        final = dest_root / vhash
        if final.exists():  # identical content already ingested
            return IngestResult(report, vhash, final, len(rows))

        build = staging / "version"
        build.mkdir()
        if root is not None:
            for r in rows:
                dst = build / "files" / r["file"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(_locate(root, r["file"])), str(dst))
        pq.write_table(pa.Table.from_pylist(sorted(rows, key=lambda r: r["sample_id"])), build / "manifest.parquet")
        build.replace(final)
        return IngestResult(report, vhash, final, len(rows))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        try:
            (dest_root / ".staging").rmdir()
        except OSError:
            pass
