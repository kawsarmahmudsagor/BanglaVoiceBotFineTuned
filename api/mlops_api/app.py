from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db import make_engine, make_sessionmaker
from .routers.catalog import router as catalog_router
from .routers.datasets import router as datasets_router
from .routers.models import router as models_router
from .routers.recipes import router as recipes_router
from .routers.runs import router as runs_router
from .routers.system import router as system_router
from .uploads import router as uploads_router
from .settings import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="Bangla Voice MLOps")
    app.state.settings = settings
    app.state.engine = make_engine(settings.database_url)
    app.state.sessionmaker = make_sessionmaker(app.state.engine)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.web_origin],
        allow_methods=["GET", "POST", "HEAD", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=[
            "Upload-Length",
            "Upload-Offset",
            "Upload-Metadata",
            "Tus-Resumable",
            "Content-Type",
        ],
        expose_headers=["Location", "Upload-Offset", "Upload-Length", "Tus-Resumable"],
    )
    app.include_router(uploads_router)
    app.include_router(datasets_router)
    app.include_router(recipes_router)
    app.include_router(runs_router)
    app.include_router(models_router)
    app.include_router(system_router)
    app.include_router(catalog_router)

    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True}

    return app
