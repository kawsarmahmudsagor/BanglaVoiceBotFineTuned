from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db import make_engine, make_sessionmaker
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

    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True}

    return app
