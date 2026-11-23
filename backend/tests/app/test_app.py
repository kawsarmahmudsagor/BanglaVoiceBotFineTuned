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


def test_migration_0003_upgrade_downgrade(tmp_path):
    db = tmp_path / "mig.db"
    cfg = Config(str(Path(__file__).parents[2] / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{db.as_posix()}")
    command.upgrade(cfg, "0002")
    eng = sa.create_engine(f"sqlite:///{db.as_posix()}")
    command.upgrade(cfg, "0003")
    insp = sa.inspect(eng)
    cols = {c["name"] for c in insp.get_columns("run")}
    assert {"parent_run_id", "parent_checkpoint"} <= cols
    assert any(fk["referred_table"] == "run" and fk["constrained_columns"] == ["parent_run_id"]
               for fk in insp.get_foreign_keys("run"))
    command.downgrade(cfg, "0002")
    cols = {c["name"] for c in sa.inspect(eng).get_columns("run")}
    assert "parent_run_id" not in cols and "parent_checkpoint" not in cols
    command.upgrade(cfg, "head")


def test_migration_0002_upgrade_downgrade(tmp_path):
    db = tmp_path / "mig.db"
    cfg = Config(str(Path(__file__).parents[2] / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{db.as_posix()}")
    command.upgrade(cfg, "0001")
    eng = sa.create_engine(f"sqlite:///{db.as_posix()}")
    with eng.begin() as c:
        c.execute(sa.text("insert into dataset (id, name, task, eval_only, created_at) "
                          "values (1, 'd', 'asr', 0, '2026-01-01')"))
        c.execute(sa.text(
            "insert into dataset_version (id, dataset_id, number, hash, path, n_samples, report, provenance, "
            "split_ratios, created_at) values (1, 1, 1, 'h', '/p', 3, '{}', '{}', '[0.9,0.05,0.05]', '2026-01-01')"))
    command.upgrade(cfg, "0002")
    with eng.connect() as c:
        assert c.execute(sa.text("select split_source, given_splits from dataset_version")).all() == [("none", "{}")]
    command.downgrade(cfg, "0001")
    cols = {c["name"] for c in sa.inspect(eng).get_columns("dataset_version")}
    assert "split_source" not in cols and "given_splits" not in cols
    command.upgrade(cfg, "head")
