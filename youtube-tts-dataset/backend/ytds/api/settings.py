"""Global settings, LLM connection test and health."""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from ..config import Settings, Thresholds
from ..llm import OllamaClient
from ..store import Store

router = APIRouter(prefix="/api")

CHECK_TIMEOUT_S = 5.0


class SettingsUpdate(BaseModel):
    thresholds: Thresholds | None = None
    cookies_path: str | None = None  # "" clears it
    llm_host: str | None = None
    llm_model: str | None = None


def llm_config(store: Store, settings: Settings) -> tuple[str, str]:
    """(host, model): the Settings page values win over the environment."""
    return (store.get_setting("llm_host") or settings.ollama_host,
            store.get_setting("llm_model") or settings.llm_model)


def make_llm(store: Store, settings: Settings, timeout: float = 120.0) -> OllamaClient:
    host, model = llm_config(store, settings)
    return OllamaClient(host, model, settings.ollama_api_key or None, timeout=timeout)


def _settings_body(request: Request) -> dict:
    store, settings = request.app.state.store, request.app.state.settings
    host, model = llm_config(store, settings)
    return {"thresholds": store.get_thresholds().model_dump(), "cookies_path": store.get_setting("cookies_path") or "",
            "llm_host": host, "llm_model": model}


@router.get("/settings")
def get_settings(request: Request) -> dict:
    return _settings_body(request)


@router.put("/settings")
def put_settings(body: SettingsUpdate, request: Request) -> dict:
    store, runner = request.app.state.store, request.app.state.runner
    if body.cookies_path:
        if not Path(body.cookies_path).is_file():
            raise HTTPException(400, f"cookies file not found: {body.cookies_path}")
    if body.thresholds is not None:
        store.set_thresholds(body.thresholds)
    if body.cookies_path is not None:
        store.set_setting("cookies_path", body.cookies_path.strip())
    llm_changed = False
    for key in ("llm_host", "llm_model"):
        value = getattr(body, key)
        if value is not None and value.strip():
            store.set_setting(key, value.strip())
            llm_changed = True
    if llm_changed:
        runner.llm = make_llm(store, request.app.state.settings)
    return _settings_body(request)


def _llm_ok(store: Store, settings: Settings) -> bool:
    try:
        return make_llm(store, settings, timeout=CHECK_TIMEOUT_S).show()
    except Exception:
        return False


@router.post("/settings/llm-test")
async def llm_test(request: Request) -> dict:
    store, settings = request.app.state.store, request.app.state.settings
    if await run_in_threadpool(_llm_ok, store, settings):
        return {"ok": True}
    host, model = llm_config(store, settings)
    return {"ok": False, "error": f"model {model} not reachable at {host}"}


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@router.get("/health")
async def health(request: Request) -> dict:
    llm_ok = await run_in_threadpool(_llm_ok, request.app.state.store, request.app.state.settings)
    return {"llm_ok": llm_ok, "cuda": _cuda(), "ffmpeg": shutil.which("ffmpeg") is not None}
