from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import func, select

from ..models import Dataset, DatasetVersion, Ingest
from .ingest import _version_hash, ingest
from .validate_audio import _sha256_file

SPLIT_RATIOS = [0.9, 0.05, 0.05]


def read_manifest(version_dir: str | Path) -> list[dict]:
    return pq.read_table(Path(version_dir) / "manifest.parquet").to_pylist()


async def _fail(session, ingest_id: int, *, error: str | None = None, report: dict | None = None) -> None:
    await session.rollback()
    row = await session.get(Ingest, ingest_id)
    if row is None:
        return
    row.state = "failed"
    row.error = error
    if report is not None:
        row.report = report
    await session.commit()


async def run_ingest(session, ingest_id: int, settings) -> None:
    """Execute one ingest request. Never raises: failures are recorded on the ingest row."""
    try:
        row = await session.get(Ingest, ingest_id)
        if row is None:
            return
        row.state = "running"
        await session.commit()

        files = [p for p in (settings.incoming_dir / row.upload_id).iterdir() if p.is_file()]
        if len(files) != 1:
            raise RuntimeError(f"expected exactly one file for upload {row.upload_id}, found {len(files)}")
        result = await asyncio.to_thread(
            ingest, files[0], row.task, settings.datasets_dir / row.dataset_name
        )
        report = asdict(result.report)
        if not result.report.ok:
            await _fail(session, ingest_id, report=report)
            return

        dataset = (await session.execute(select(Dataset).where(Dataset.name == row.dataset_name))).scalar_one_or_none()
        if dataset is None:
            dataset = Dataset(name=row.dataset_name, task=row.task, eval_only=row.eval_only)
            session.add(dataset)
            await session.flush()
        top = (
            await session.execute(
                select(func.max(DatasetVersion.number)).where(DatasetVersion.dataset_id == dataset.id)
            )
        ).scalar()
        version = DatasetVersion(
            dataset_id=dataset.id,
            number=(top or 0) + 1,
            hash=result.version_hash,
            path=str(result.version_dir),
            n_samples=result.n_samples,
            report=report,
            provenance=row.provenance,
            split_ratios=SPLIT_RATIOS,
        )
        session.add(version)
        await session.flush()
        row.state = "done"
        row.report = report
        row.version_id = version.id
        await session.commit()
    except Exception as exc:  # noqa: BLE001 - run_ingest must never raise
        try:
            await _fail(session, ingest_id, error=str(exc))
        except Exception:  # noqa: BLE001
            pass


class SampleNotFound(Exception):
    pass


class EditNotSupported(Exception):
    pass


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


async def edit_transcript(session, version: DatasetVersion, sample_id: str, text: str, settings) -> DatasetVersion:
    dataset = await session.get(Dataset, version.dataset_id)
    if dataset.task == "llm-sft":
        raise EditNotSupported("transcript edits apply to asr/tts datasets only")
    src_dir = Path(version.path)
    rows = [{k: v for k, v in r.items() if v is not None} for r in read_manifest(src_dir)]
    idx = next((i for i, r in enumerate(rows) if r["sample_id"] == sample_id), None)
    if idx is None:
        raise SampleNotFound(sample_id)

    row = rows[idx]
    audio = src_dir / "files" / row["file"]
    row["text"] = text
    row["sample_id"] = hashlib.sha256((text + "\n" + _sha256_file(audio)).encode("utf-8")).hexdigest()[:16]
    # same rule as validate_audio; split is deliberately kept
    vhash = _version_hash(rows)[:16]

    dest_root = settings.datasets_dir / dataset.name
    final = dest_root / vhash
    if not final.exists():
        staging = dest_root / ".staging" / uuid.uuid4().hex
        try:
            build = staging / "version"
            for r in rows:
                _link_or_copy(src_dir / "files" / r["file"], build / "files" / r["file"])
            ordered = sorted(rows, key=lambda r: r["sample_id"])
            columns = list(dict.fromkeys(k for r in ordered for k in r))
            pq.write_table(pa.table({c: [r.get(c) for r in ordered] for c in columns}),
                           build / "manifest.parquet")
            build.replace(final)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            try:
                (dest_root / ".staging").rmdir()
            except OSError:
                pass

    top = (
        await session.execute(select(func.max(DatasetVersion.number)).where(DatasetVersion.dataset_id == dataset.id))
    ).scalar()
    child = DatasetVersion(
        dataset_id=dataset.id,
        number=(top or 0) + 1,
        hash=vhash,
        path=str(final),
        n_samples=len(rows),
        report=version.report,
        provenance=version.provenance,
        split_ratios=version.split_ratios,
        parent_version_id=version.id,
    )
    session.add(child)
    await session.commit()
    return child
