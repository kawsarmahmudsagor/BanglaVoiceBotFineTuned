"""FastAPI app factory: REST routes, the job WebSocket and the built frontend."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ..config import SUB_ROOT, Settings, export_hf_token
from ..events import EventBus
from ..store import Store
from . import dataset, jobs, review, settings as settings_routes, ws

FRONTEND_DIST = SUB_ROOT / "frontend" / "dist"


def _default_runner(settings: Settings, store: Store, bus: EventBus):
    from .. import models
    from ..gpu import GpuWorker
    from ..pipeline import quality, separate
    from ..runner import JobRunner, StageFns

    gpu = GpuWorker({
        "demucs": separate.load_demucs,
        "ecapa": quality.load_ecapa,
        "model_a": models.loader("a", store, settings),
        "model_b": models.loader("b", store, settings),
    })
    stages = StageFns.default()
    download = stages.download

    def download_with_cookies(video_id, url, out_dir, on_progress):
        cookies = store.get_setting("cookies_path")
        return download(video_id, url, out_dir, on_progress, cookies=Path(cookies) if cookies else None)

    stages.download = download_with_cookies
    llm = settings_routes.make_llm(store, settings)
    return JobRunner(store, bus, gpu, llm, settings, stages)


def create_app(settings: Settings, runner=None, store: Store | None = None, bus: EventBus | None = None,
               ) -> FastAPI:
    export_hf_token(settings)  # before any model loader can reach huggingface_hub
    store = store or Store(Path(settings.data_dir) / "ytds.db")
    bus = bus or EventBus()
    runner = runner or _default_runner(settings, store, bus)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        bus.bind_loop(asyncio.get_running_loop())
        runner.start()
        try:
            yield
        finally:
            runner.stop()

    app = FastAPI(title="ytds", lifespan=lifespan)
    app.state.settings, app.state.store, app.state.bus, app.state.runner = settings, store, bus, runner
    app.state.exports = set()  # dataset ids with an export running
    for module in (jobs, review, dataset, settings_routes, ws):
        app.include_router(module.router)
    if FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
    return app
