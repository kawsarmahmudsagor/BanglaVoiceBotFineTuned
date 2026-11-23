"""Prepared-data cache contract shared by every training backend.

Importable without pyarrow; it is imported lazily inside iter_rows.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path

from pydantic import BaseModel

from .contract import DatasetRef, RunConfig


class PrepareError(RuntimeError):
    pass


class PrepSpec(BaseModel):
    family: str
    version: int
    params: dict

    def hash(self) -> str:
        blob = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class Skip(BaseModel):
    sample_id: str
    split: str
    reason: str


class BuildResult(BaseModel):
    counts: dict[str, int]
    skipped: list[Skip] = []
    files: list[str] = []


class PrepInfo(BaseModel):
    spec: PrepSpec
    spec_hash: str
    dataset_hash: str
    counts: dict[str, int]
    skipped: list[Skip]
    files: list[str]
    built_at: float
    build_s: float
    path: str  # set on return, not trusted from disk
    cache_hit: bool = False  # set on return, not trusted from disk


def iter_rows(ds: DatasetRef, split: str) -> Iterator[dict]:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(ds.manifest)
    table = table.filter(pc.equal(table["split"], split))
    for row in table.to_pylist():
        if "messages_json" in row:
            row["messages"] = json.loads(row.pop("messages_json"))
            tools = row.pop("tools_json", None)
            row["tools"] = json.loads(tools) if tools else None
        if "file" in row:
            row["audio_path"] = str(Path(ds.root) / "files" / row["file"])
        yield row


def load_splits(ds: DatasetRef, splits: list[str]) -> dict[str, list[dict]]:
    return {s: list(iter_rows(ds, s)) for s in splits}


def prepared_dir(cfg: RunConfig, ds: DatasetRef, spec: PrepSpec) -> Path:
    return Path(cfg.prepared_root) / ds.hash / spec.hash()


class _FileLock:
    """Exclusive inter-process lock on a file (fcntl on POSIX, msvcrt on Windows)."""

    def __init__(self, path: Path):
        self.path = path
        self._f = None

    def __enter__(self) -> "_FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a+b")
        if sys.platform == "win32":
            import msvcrt

            self._f.seek(0)
            while True:
                try:
                    msvcrt.locking(self._f.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:  # LK_LOCK gives up after ~10 s; keep waiting
                    continue
        else:
            import fcntl

            fcntl.flock(self._f.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc) -> None:
        assert self._f is not None
        try:
            if sys.platform == "win32":
                import msvcrt

                self._f.seek(0)
                msvcrt.locking(self._f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._f.fileno(), fcntl.LOCK_UN)
        finally:
            self._f.close()


def _read_hit(target: Path) -> PrepInfo | None:
    try:
        data = json.loads((target / "prepared.json").read_bytes())
        data.update(path=str(target), cache_hit=True)
        return PrepInfo.model_validate(data)
    except (FileNotFoundError, ValueError):
        return None


def ensure_prepared(
    cfg: RunConfig,
    ds: DatasetRef,
    spec: PrepSpec,
    build: Callable[[dict[str, list[dict]], Path], BuildResult],
    *,
    splits: list[str] | None = None,
    max_skip_share: float = 0.05,
) -> PrepInfo:
    target = prepared_dir(cfg, ds, spec)
    hit = _read_hit(target)
    if hit:
        return hit
    with _FileLock(target.with_name(target.name + ".lock")):
        hit = _read_hit(target)
        if hit:
            return hit
        for stale in target.parent.glob(target.name + ".tmp-*"):
            shutil.rmtree(stale, ignore_errors=True)
        tmp = target.with_name(f"{target.name}.tmp-{os.getpid()}")
        tmp.mkdir(parents=True)
        try:
            t0 = time.time()
            rows = load_splits(ds, splits if splits is not None else ds.splits)
            total = sum(len(v) for v in rows.values())
            result = build(rows, tmp)
            share = len(result.skipped) / max(1, total)
            if share > max_skip_share:
                top3 = "; ".join(f"{r} ×{n}" for r, n in Counter(s.reason for s in result.skipped).most_common(3))
                raise PrepareError(f"{share * 100:.1f}% of rows skipped (limit {max_skip_share * 100:.1f}%): {top3}")
            now = time.time()
            info = PrepInfo(spec=spec, spec_hash=spec.hash(), dataset_hash=ds.hash, counts=result.counts,
                            skipped=result.skipped, files=result.files, built_at=now, build_s=now - t0,
                            path=str(target))
            (tmp / "prepared.json").write_bytes(
                info.model_dump_json(indent=2, exclude={"path", "cache_hit"}).encode("utf-8"))
            if target.exists():
                shutil.rmtree(target)
            os.replace(tmp, target)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return info
