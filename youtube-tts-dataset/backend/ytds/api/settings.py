"""Global settings, LLM connection test and health."""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from .. import models as model_registry
from ..config import Settings, Thresholds
from ..llm import OllamaClient
from ..store import Store

router = APIRouter(prefix="/api")

CHECK_TIMEOUT_S = 5.0
DNSMOS_MODEL = "DNSMOS P.835 OVRL (torchmetrics)"


class SettingsUpdate(BaseModel):
    thresholds: Thresholds | None = None
    cookies_path: str | None = None  # "" clears it
    llm_host: str | None = None
    llm_model: str | None = None
    model_a: str | None = None  # transcription option ids (spec 2.2)
    model_b: str | None = None


def llm_config(store: Store, settings: Settings) -> tuple[str, str]:
    """(host, model): the Settings page values win over the environment."""
    return (store.get_setting("llm_host") or settings.ollama_host,
            store.get_setting("llm_model") or settings.llm_model)


def make_llm(store: Store, settings: Settings, timeout: float = 120.0) -> OllamaClient:
    host, model = llm_config(store, settings)
    return OllamaClient(host, model, settings.ollama_api_key or None, timeout=timeout)


def stage_models(llm_model: str, store: Store) -> dict:
    """Read-only list of the model behind each stage (spec section 4.4); model A/B are the current choices."""
    from dataprep.steps.speakers import MODEL as SPEAKER_MODEL

    from ..exporter import _model_ids

    ids = _model_ids()
    label = {o.id: o.label for o in model_registry.OPTIONS}
    return {"separate": ids["separation"], "vad": ids["vad"], "dnsmos": DNSMOS_MODEL, "speaker": SPEAKER_MODEL,
            "model_a": label[model_registry.current(store, "a")],
            "model_b": label[model_registry.current(store, "b")], "llm": llm_model}


def transcription_choices(store: Store, settings: Settings) -> dict:
    out = {}
    for slot in ("a", "b"):
        opts = []
        for o in model_registry.options(slot):
            ok, expected = model_registry.available(o.id, settings)
            opts.append({"id": o.id, "label": o.label, "available": ok, "expected": expected})
        out[slot] = {"current": model_registry.current(store, slot), "options": opts}
    return out


def _check_model(slot: str, option_id: str, settings: Settings) -> None:
    o = model_registry.BY_ID.get(option_id)
    if o is None or o.slot != slot:
        raise HTTPException(422, f"unknown model {slot.upper()} option: {option_id}")
    ok, expected = model_registry.available(option_id, settings)
    if not ok:
        raise HTTPException(422, f"model {option_id} is not available: folder not found: {expected}")


def _settings_body(request: Request) -> dict:
    store, settings = request.app.state.store, request.app.state.settings
    host, model = llm_config(store, settings)
    return {"thresholds": store.get_thresholds().model_dump(), "cookies_path": store.get_setting("cookies_path") or "",
            "llm_host": host, "llm_model": model, "models": stage_models(model, store),
            "transcription": transcription_choices(store, settings)}


@router.get("/settings")
def get_settings(request: Request) -> dict:
    return _settings_body(request)


@router.put("/settings")
def put_settings(body: SettingsUpdate, request: Request) -> dict:
    store, runner, settings = request.app.state.store, request.app.state.runner, request.app.state.settings
    if body.cookies_path:
        if not Path(body.cookies_path).is_file():
            raise HTTPException(400, f"cookies file not found: {body.cookies_path}")
    for slot in ("a", "b"):  # validate before anything is stored
        value = getattr(body, f"model_{slot}")
        if value is not None:
            _check_model(slot, value, settings)
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
        runner.llm = make_llm(store, settings)
    for slot in ("a", "b"):
        value = getattr(body, f"model_{slot}")
        if value is None or value == model_registry.current(store, slot):
            continue
        store.set_setting(f"model_{slot}", value)
        gpu = getattr(runner, "gpu", None)
        if gpu is not None:  # the next transcribe stage loads the new choice
            gpu.unload(f"model_{slot}")
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
    cuda = await run_in_threadpool(_cuda)
    return {"llm_ok": llm_ok, "cuda": cuda, "ffmpeg": shutil.which("ffmpeg") is not None}
