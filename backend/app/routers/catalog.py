"""Read-only model catalog, loaded from YAML files on each request."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request

from app.core.catalog import catalog_summary, load_catalog

router = APIRouter(prefix="/api")


@router.get("/catalog")
async def list_catalog(request: Request, task: Literal["llm-sft", "asr", "tts"] | None = None) -> dict:
    entries, errors = load_catalog(request.app.state.settings.catalog_dir)
    return {
        "entries": [catalog_summary(e) for e in entries if task is None or e.task == task],
        "errors": [e.model_dump() for e in errors],
    }


@router.get("/catalog/{entry_id}")
async def get_catalog_entry(entry_id: str, request: Request) -> dict:
    entries, _ = load_catalog(request.app.state.settings.catalog_dir)
    for e in entries:
        if e.id == entry_id:
            return e.model_dump(mode="json")
    raise HTTPException(404, f"catalog entry '{entry_id}' not found")
