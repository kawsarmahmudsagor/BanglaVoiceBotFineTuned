import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from pathlib import Path

from app.db import Base


async def test_health(client):
    assert (await client.get("/api/health")).json() == {"ok": True}


async def test_cors_allows_web_origin_only(client):
    r = await client.options("/api/health", headers={"Origin": "http://localhost:3000",
                                                     "Access-Control-Request-Method": "GET"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"
    r = await client.options("/api/health", headers={"Origin": "http://evil",
                                                     "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in r.headers


def test_alembic_head_matches_models(tmp_path):
    db = tmp_path / "mig.db"
    cfg = Config(str(Path(__file__).parents[2] / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{db.as_posix()}")
    command.upgrade(cfg, "head")
    eng = sa.create_engine(f"sqlite:///{db.as_posix()}")
    insp = sa.inspect(eng)
    assert set(insp.get_table_names()) - {"alembic_version"} == set(Base.metadata.tables)
    with eng.connect() as c:
        assert c.execute(sa.text("select id, holder_id from gpu_lock")).all() == [(1, None)]
