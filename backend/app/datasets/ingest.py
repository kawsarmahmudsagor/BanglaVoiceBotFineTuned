from __future__ import annotations

import hashlib
import json
import os
import shutil
import unicodedata
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.parquet as pq

from .report import Issue, Report
from .validate_audio import _AUDIO_EXT, _META_SPLIT_RE, metadata_files, validate_audio
from .validate_llm import validate_llm

_TASK_EXT = {"llm-sft": (".jsonl", ".zip"), "asr": (".zip",), "tts": (".zip",)}
GB = 1024**3


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


def _extract(upload: Path, dest: Path, report: Report, disk_min_gb: float = 0) -> Path | None:
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
        need = sum(i.file_size for i, _ in members)
        room = shutil.disk_usage(dest).free - disk_min_gb * GB
        if need > room:
            report.errors.append(Issue(
                "insufficient_disk",
                f"archive unpacks to {need / GB:.1f} GB but only {max(room, 0) / GB:.1f} GB is free "
                f"above the {disk_min_gb:g} GB reserve",
            ))
            return None
        for info, name in members:
            if name.endswith("/") or info.is_dir():
                continue
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
    def has_meta(d: Path) -> bool:
        plain, per_split, problems = metadata_files(d)
        return plain is not None or bool(per_split) or bool(problems)

    if has_meta(dest):
        return dest
    tops = [p for p in dest.iterdir() if p.name != "__MACOSX"]
    if len(tops) == 1 and tops[0].is_dir() and has_meta(tops[0]):
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


def _publish(report: Report, rows: list[dict], dest_root: Path, staging: Path, transfer, disk_check=None) -> IngestResult:
    """Everything after validation: hash, disk check, version dir, manifest.

    `transfer(row, dst)` places one audio file (None for llm-sft); `disk_check(rows)` returns an Issue when
    the files that must be copied do not fit (None skips the check).
    """
    if report.ok and not rows:
        report.errors.append(Issue("no_samples", "upload contains no samples"))
    if not report.ok:
        return IngestResult(report, None, None, 0)

    given = {"train": 0, "val": 0, "test": 0, "unassigned": 0}
    for r in rows:
        given[r.get("split") or "unassigned"] += 1
    report.stats["given_splits"] = given
    report.stats["split_source"] = "dataset" if given["unassigned"] < len(rows) else "none"
    report.stats["n_samples"] = len(rows)

    vhash = _version_hash(rows)[:16]
    final = dest_root / vhash
    if final.exists():  # identical content already ingested
        return IngestResult(report, vhash, final, len(rows))

    if disk_check is not None:
        issue = disk_check(rows)
        if issue is not None:
            report.errors.append(issue)
            return IngestResult(report, None, None, 0)

    build = staging / "version"
    build.mkdir()
    if transfer is not None:
        for r in rows:
            dst = build / "files" / r["file"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            transfer(r, dst)
    ordered = sorted(rows, key=lambda r: r["sample_id"])
    columns = list(dict.fromkeys(k for r in ordered for k in r))  # union of keys, first-seen order
    table = pa.table({c: [r.get(c) for r in ordered] for c in columns})
    pq.write_table(table, build / "manifest.parquet")
    build.replace(final)
    return IngestResult(report, vhash, final, len(rows))


def ingest(
    upload_file: Path, task: str, dest_root: Path, disk_min_gb: float = 0, workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> IngestResult:
    upload_file = Path(upload_file)
    dest_root = Path(dest_root)
    want = _TASK_EXT.get(task)
    if want is None or upload_file.suffix.lower() not in want:
        msg = f"unknown task '{task}'" if want is None else f"task '{task}' expects a {' or '.join(want)} upload"
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
            root = _extract(upload_file, extracted, report, disk_min_gb)
            if root is not None:
                report, rows = validate_audio(root, task, workers=workers, progress=progress)
        if root is None:
            return _publish(report, rows, dest_root, staging, None)
        return _publish(report, rows, dest_root, staging,
                        lambda r, dst: shutil.move(str(_locate(root, r["file"])), str(dst)))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        try:
            (dest_root / ".staging").rmdir()
        except OSError:
            pass


def _stage_dir(src: Path, staged: Path, dest_root: Path, disk_min_gb: float, report: Report) -> bool:
    """Hard-link (or copy) the dataset's metadata and audio files from `src` into `staged`.

    Returns False after recording an error. Every file is checked to resolve inside `src` before anything is
    linked. Metadata files are always copied (small; linking would let the version share their mode). Audio is
    hard-linked when `src` and `dest_root` share a device; a failed link falls back to a copy only after a disk
    check for that file. Across devices every byte is copied, so the total is checked up front.
    """
    real_src = src.resolve()
    meta: list[Path] = []
    audio: list[Path] = []
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(src)
        if p.suffix.lower() in _AUDIO_EXT:
            audio.append(rel)
        elif len(rel.parts) == 1 and (p.name == "metadata.csv" or _META_SPLIT_RE.match(p.name)):
            meta.append(rel)
    for rel in [*meta, *audio]:
        if not (src / rel).resolve().is_relative_to(real_src):
            report.errors.append(Issue(
                "unsafe_path", f"'{rel.as_posix()}' is a symlink that resolves outside the dataset directory"))
            return False

    def room() -> float:
        return shutil.disk_usage(staged).free - disk_min_gb * GB

    def no_room(need: int) -> bool:
        report.errors.append(Issue(
            "insufficient_disk",
            f"dataset needs {need / GB:.1f} GB of copies but only {max(room(), 0) / GB:.1f} GB is free "
            f"above the {disk_min_gb:g} GB reserve",
        ))
        return False

    same_dev = os.stat(src).st_dev == os.stat(dest_root).st_dev
    size = {rel: (src / rel).stat().st_size for rel in [*meta, *audio]}
    need = sum(size[r] for r in meta) + (0 if same_dev else sum(size[r] for r in audio))
    if need > room():
        return no_room(need)
    for rel in meta:
        shutil.copyfile(src / rel, staged / rel)
    for rel in audio:
        dst = staged / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if same_dev:
            try:
                os.link(src / rel, dst)
                continue
            except OSError:
                if size[rel] > room():  # the link failed: this file must be copied after all
                    return no_room(size[rel])
        shutil.copy2(src / rel, dst)
    return True


def _move_ro(src: Path, dst: Path) -> None:
    os.replace(src, dst)
    os.chmod(dst, 0o444)  # version files are read-only (a hard-linked source shares the mode, by design)


def ingest_dir(
    src: Path, task: str, dest_root: Path, disk_min_gb: float = 0, workers: int = 1,
    progress: Callable[[int, int], None] | None = None, *, _on_staged: Callable[[Path], None] | None = None,
) -> IngestResult:
    """Ingest an audio dataset directory in place: stage it, validate the staged copy, then publish.

    Every file is hard-linked (or copied) into the staging dir first, and the staged files are what gets
    validated, hashed and stored, so a re-export of `src` during a long validation cannot make the manifest
    describe other content than the version holds. Nothing under `src` is modified, moved or deleted (a
    hard-linked audio file shares the version's read-only mode). `_on_staged(staged_dir)` is a test hook.
    """
    src = Path(src)
    dest_root = Path(dest_root)
    if task not in ("asr", "tts"):
        return IngestResult(Report(errors=[Issue(
            "unsupported_format", f"local-path ingest supports asr and tts, not '{task}'")]), None, None, 0)

    staging = dest_root / ".staging" / uuid.uuid4().hex
    staging.mkdir(parents=True)
    try:
        staged = staging / "src"
        staged.mkdir()
        report = Report()
        if not _stage_dir(src, staged, dest_root, disk_min_gb, report):
            return IngestResult(report, None, None, 0)
        if _on_staged is not None:
            _on_staged(staged)
        report, rows = validate_audio(staged, task, workers=workers, progress=progress)
        return _publish(report, rows, dest_root, staging,
                        lambda r, dst: _move_ro(_locate(staged, r["file"]), dst))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        try:
            (dest_root / ".staging").rmdir()
        except OSError:
            pass


class ImportPathError(Exception):
    pass


def resolve_import_path(raw: str, import_root: Path) -> Path:
    """Resolve `raw` (relative to import_root, or absolute) to a directory inside import_root.

    Raises ImportPathError when it escapes the root (including via `..` or symlinks) or is not a directory,
    and FileNotFoundError when it is inside the root but missing. A missing path outside the root is an
    ImportPathError, so existence outside the root is never disclosed.
    """
    denied = ImportPathError("path is not allowed")  # one generic message: never echo the path or errno
    root = Path(import_root).resolve()
    missing = False
    try:
        p = Path(raw)
        cand = p if p.is_absolute() else root / p
        try:
            resolved = cand.resolve(strict=True)
        except (FileNotFoundError, NotADirectoryError):
            # lexically normalised check so a missing path outside the root still reads as an escape
            if not Path(os.path.normpath(cand)).is_relative_to(root):
                raise denied from None
            missing = True
        else:
            if not resolved.is_relative_to(root) or not resolved.is_dir():
                raise denied
    except ImportPathError:
        raise
    except (OSError, ValueError, RuntimeError):  # NUL bytes, loops, permissions, invalid names
        raise denied from None
    if missing:
        raise FileNotFoundError("directory not found")
    return resolved
