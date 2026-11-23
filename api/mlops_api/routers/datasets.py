from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..datasets.versions import EditNotSupported, SampleNotFound, edit_transcript, read_manifest
from ..db import get_session
from ..models import Dataset, DatasetVersion, Ingest, Upload

router = APIRouter(prefix="/api")

SLUG = r"^[a-z0-9]+(-[a-z0-9]+)*$"


def iso(d: dt.datetime | None) -> str | None:
    if d is None:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.isoformat()


class Provenance(BaseModel):
    source: str = ""
    license: str = ""
    consent: str = ""
    commercial_ok: bool = False
    notes: str = ""


class IngestRequest(BaseModel):
    upload_id: str
    name: str = Field(pattern=SLUG)
    task: Literal["llm-sft", "asr", "tts"]
    eval_only: bool = False
    provenance: Provenance


class EditRequest(BaseModel):
    sample_id: str
    text: str


class DatasetPatch(BaseModel):
    eval_only: bool


async def _dataset(session: AsyncSession, name: str) -> Dataset:
    ds = (await session.execute(select(Dataset).where(Dataset.name == name))).scalar_one_or_none()
    if ds is None:
        raise HTTPException(404, f"dataset '{name}' not found")
    return ds


async def _versions(session: AsyncSession, ds: Dataset) -> list[DatasetVersion]:
    q = select(DatasetVersion).where(DatasetVersion.dataset_id == ds.id).order_by(DatasetVersion.number)
    return list((await session.execute(q)).scalars())


async def _version(session: AsyncSession, name: str, n: int) -> tuple[Dataset, DatasetVersion]:
    ds = await _dataset(session, name)
    v = (
        await session.execute(
            select(DatasetVersion).where(DatasetVersion.dataset_id == ds.id, DatasetVersion.number == n)
        )
    ).scalar_one_or_none()
    if v is None:
        raise HTTPException(404, f"version {n} of '{name}' not found")
    return ds, v


def _summary(v: DatasetVersion, parent_number: int | None = None) -> dict:
    return {
        "id": v.id,
        "number": v.number,
        "hash": v.hash,
        "n_samples": v.n_samples,
        "commercial_ok": bool((v.provenance or {}).get("commercial_ok", False)),
        "parent_number": parent_number,
        "created_at": iso(v.created_at),
    }


def _dataset_out(ds: Dataset) -> dict:
    return {"id": ds.id, "name": ds.name, "task": ds.task, "eval_only": ds.eval_only,
            "created_at": iso(ds.created_at)}


async def _parent_number(session: AsyncSession, v: DatasetVersion) -> int | None:
    if v.parent_version_id is None:
        return None
    parent = await session.get(DatasetVersion, v.parent_version_id)
    return parent.number if parent else None


@router.post("/datasets/ingest", status_code=202)
async def create_ingest(body: IngestRequest, session: AsyncSession = Depends(get_session)) -> dict:
    upload = await session.get(Upload, body.upload_id)
    if upload is None:
        raise HTTPException(404, "upload not found")
    if upload.state != "complete":
        raise HTTPException(409, "upload is not complete")
    existing = (await session.execute(select(Dataset).where(Dataset.name == body.name))).scalar_one_or_none()
    if existing is not None and existing.task != body.task:
        raise HTTPException(409, f"dataset '{body.name}' is task '{existing.task}'")
    row = Ingest(
        upload_id=body.upload_id,
        dataset_name=body.name,
        task=body.task,
        eval_only=body.eval_only,
        provenance=body.provenance.model_dump(),
    )
    session.add(row)
    await session.commit()
    return {"ingest_id": row.id}


@router.get("/ingests/{ingest_id}")
async def get_ingest(ingest_id: int, session: AsyncSession = Depends(get_session)) -> dict:
    row = await session.get(Ingest, ingest_id)
    if row is None:
        raise HTTPException(404, "ingest not found")
    version = None
    if row.version_id is not None:
        v = await session.get(DatasetVersion, row.version_id)
        version = _summary(v, await _parent_number(session, v))
    return {"id": row.id, "state": row.state, "report": row.report, "version": version, "error": row.error,
            "dataset": row.dataset_name, "created_at": iso(row.created_at)}


@router.get("/datasets")
async def list_datasets(session: AsyncSession = Depends(get_session)) -> list[dict]:
    out = []
    for ds in (await session.execute(select(Dataset).order_by(Dataset.name))).scalars():
        versions = await _versions(session, ds)
        latest = None
        if versions:
            v = versions[-1]
            latest = {"number": v.number, "n_samples": v.n_samples,
                      "commercial_ok": bool((v.provenance or {}).get("commercial_ok", False)),
                      "created_at": iso(v.created_at)}
        out.append({**_dataset_out(ds), "latest": latest})
    return out


@router.get("/datasets/{name}")
async def get_dataset(name: str, session: AsyncSession = Depends(get_session)) -> dict:
    ds = await _dataset(session, name)
    versions = await _versions(session, ds)
    numbers = {v.id: v.number for v in versions}
    return {**_dataset_out(ds), "versions": [_summary(v, numbers.get(v.parent_version_id)) for v in versions]}


@router.patch("/datasets/{name}")
async def patch_dataset(name: str, body: DatasetPatch, session: AsyncSession = Depends(get_session)) -> dict:
    ds = await _dataset(session, name)
    ds.eval_only = body.eval_only
    await session.commit()
    return _dataset_out(ds)


@router.get("/datasets/{name}/versions/{n}")
async def get_version(name: str, n: int, session: AsyncSession = Depends(get_session)) -> dict:
    ds, v = await _version(session, name, n)
    splits = {"train": 0, "val": 0, "test": 0}
    for r in read_manifest(v.path):
        splits[r["split"]] = splits.get(r["split"], 0) + 1
    return {**_summary(v, await _parent_number(session, v)), "dataset": ds.name, "task": ds.task,
            "report": v.report, "provenance": v.provenance, "split_ratios": v.split_ratios, "splits": splits}


@router.get("/datasets/{name}/versions/{n}/samples")
async def get_samples(
    name: str, n: int, offset: int = 0, limit: int = 50, split: str | None = None,
    warning: str | None = None, session: AsyncSession = Depends(get_session),
) -> dict:
    ds, v = await _version(session, name, n)
    rows = read_manifest(v.path)
    if split:
        rows = [r for r in rows if r["split"] == split]
    if warning:
        report = v.report or {}
        ids = {i["sample_id"] for i in report.get("warnings", []) + report.get("errors", [])
               if i.get("code") == warning and i.get("sample_id")}
        rows = [r for r in rows if r["sample_id"] in ids]
    offset = max(0, offset)
    limit = max(1, min(limit, 500))
    return {"total": len(rows), "rows": rows[offset: offset + limit]}


@router.get("/datasets/{name}/versions/{n}/audio/{sample_id}")
async def get_audio(
    name: str, n: int, sample_id: str, session: AsyncSession = Depends(get_session)
) -> FileResponse:
    ds, v = await _version(session, name, n)
    if ds.task == "llm-sft":
        raise HTTPException(404, "llm datasets have no audio")
    row = next((r for r in read_manifest(v.path) if r["sample_id"] == sample_id), None)
    if row is None:
        raise HTTPException(404, "sample not found")
    root = (Path(v.path) / "files").resolve()
    path = (root / row["file"]).resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(404, "audio file not found")
    return FileResponse(path)


@router.post("/datasets/{name}/versions/{n}/edits", status_code=201)
async def post_edit(
    name: str, n: int, body: EditRequest, request: Request, session: AsyncSession = Depends(get_session)
) -> dict:
    ds, v = await _version(session, name, n)
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "text must not be empty")
    try:
        child = await edit_transcript(session, v, body.sample_id, text, request.app.state.settings)
    except EditNotSupported as e:
        raise HTTPException(400, str(e)) from e
    except SampleNotFound as e:
        raise HTTPException(404, "sample not found") from e
    return _summary(child, v.number)
