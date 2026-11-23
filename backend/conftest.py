import sys
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.app import create_app
from app.db import Base, make_sessionmaker
from app import models
from app.settings import Settings

REPO = Path(__file__).parent


@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    monkeypatch.setenv("BACKEND_PYTHON_FAKE", sys.executable)
    data_root = tmp_path / "root"
    data_root.mkdir()
    return Settings(
        _env_file=None,  # never pick up a developer's .env
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'test.db').as_posix()}",
        data_root=data_root,
        backends_root=REPO / "workers" / "backends",
        recipes_dir=tmp_path / "recipes",
        catalog_dir=tmp_path / "catalog",
    )


@pytest_asyncio.fixture
async def session_factory(settings):
    engine = create_async_engine(settings.database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = make_sessionmaker(engine)
    async with factory() as s:
        s.add(models.GpuLock(id=1))
        await s.commit()
    yield factory
    await engine.dispose()


@pytest_asyncio.fixture
async def session(session_factory):
    async with session_factory() as s:
        yield s


@pytest_asyncio.fixture
async def client(settings, session_factory):
    app = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    await app.state.engine.dispose()
