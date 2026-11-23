import pytest
from sqlalchemy import select

from app.core.backend_kit import mark_complete, write_init_meta
from app.core.contract import RunDir
from app.core.recipes import RecipeSpec
from app.models import ModelVersion, Recipe, RecipeVersion, Run
from app.runs_init import link_tree, resolve_init

from runs_helpers import seed_dataset, spec as base_spec


def _spec(**over) -> RecipeSpec:
    return RecipeSpec.model_validate({**base_spec(name="child"), **over})


def _checkpoint(run_dir, name, method="none", step=5):
    d = RunDir(run_dir).checkpoints_dir / name
    d.mkdir(parents=True)
    (d / "weights.txt").write_text("w")
    write_init_meta(d, backend="fake", base_model="m", method=method, format="fake", step=step)
    mark_complete(d)
    return d


@pytest.fixture
async def done_run(session, tmp_path):
    rec = Recipe(name="parent")
    session.add(rec)
    await session.flush()
    rv = RecipeVersion(recipe_id=rec.id, number=1, spec=base_spec(name="parent"), yaml="x", frozen=True)
    session.add(rv)
    await session.flush()
    run = Run(id="r-parent01", recipe_version_id=rv.id, state="done", run_dir=str(tmp_path / "parent"),
              commercial_ok=True)
    session.add(run)
    await session.commit()
    _checkpoint(run.run_dir, "final")
    return run


async def test_resolve_init_none_without_init_from(session, settings):
    assert await resolve_init(session, _spec(), settings) == (None, [])


async def test_resolve_init_ok(session, settings, done_run):
    res, errors = await resolve_init(session, _spec(init_from={"run": done_run.id}), settings)
    assert errors == []
    run, path, ref = res
    assert run.id == done_run.id and path == RunDir(done_run.run_dir).checkpoints_dir / "final"
    assert (ref.source_run, ref.checkpoint, ref.method, ref.format) == (done_run.id, "final", "none", "fake")


async def test_resolve_init_run_not_found(session, settings):
    res, errors = await resolve_init(session, _spec(init_from={"run": "r-nope"}), settings)
    assert res is None and errors == ["init_from: run r-nope not found"]


async def test_resolve_init_rejects_backend_mismatch(session, settings, done_run):
    res, errors = await resolve_init(session, _spec(backend="other", init_from={"run": done_run.id}), settings)
    assert errors == [f"init_from: run {done_run.id} is backend 'fake', recipe is backend 'other'"]


async def test_resolve_init_rejects_task_and_base_model_mismatch(session, settings, done_run):
    _, errors = await resolve_init(
        session, _spec(task="asr", base_model="z", init_from={"run": done_run.id}), settings)
    assert errors == [f"init_from: run {done_run.id} is task 'llm-sft', recipe is task 'asr'",
                      f"init_from: run {done_run.id} is base_model 'm', recipe is base_model 'z'"]


async def test_resolve_init_rejects_method_pair(session, settings, done_run):
    d = RunDir(done_run.run_dir).checkpoints_dir / "final"
    write_init_meta(d, backend="fake", base_model="m", method="adapter", format="x", step=1)
    _, errors = await resolve_init(session, _spec(method={"type": "full"}, init_from={"run": done_run.id}), settings)
    assert errors == ["init_from: method 'adapter' cannot start a 'full' run"]


async def test_resolve_init_model_version_uses_final(session, settings, done_run):
    session.add(ModelVersion(name="m1", number=2, run_id=done_run.id, task="llm-sft", base_model="m"))
    await session.commit()
    res, errors = await resolve_init(session, _spec(init_from={"model_version": "m1@v2"}), settings)
    assert errors == [] and res[2].checkpoint == "final"
    _, errors = await resolve_init(session, _spec(init_from={"model_version": "m1@v9"}), settings)
    assert errors == ["init_from: model version m1@v9 not found"]


async def test_resolve_init_missing_checkpoint(session, settings, done_run):
    _, errors = await resolve_init(session, _spec(init_from={"run": done_run.id, "checkpoint": "latest"}), settings)
    assert errors == [f"init_from: run {done_run.id} has no complete checkpoint 'latest'"]
    _checkpoint(done_run.run_dir, "checkpoint-300", step=300)
    res, errors = await resolve_init(session, _spec(init_from={"run": done_run.id, "checkpoint": 300}), settings)
    assert errors == [] and res[2].checkpoint == "checkpoint-300"


def test_link_tree_survives_source_delete(tmp_path):
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text("a")
    (src / "sub" / "b.txt").write_text("b")
    dst = tmp_path / "dst"
    link_tree(src, dst)
    assert not (tmp_path / "dst.tmp").exists()
    for p in sorted(src.rglob("*"), reverse=True):
        p.unlink() if p.is_file() else p.rmdir()
    src.rmdir()
    assert (dst / "a.txt").read_text() == "a" and (dst / "sub" / "b.txt").read_text() == "b"


def test_link_tree_falls_back_to_copy(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("a")

    def boom(*a, **k):
        raise OSError("cross-device")
    monkeypatch.setattr("app.runs_init.os.link", boom)
    link_tree(src, tmp_path / "dst")
    assert (tmp_path / "dst" / "a.txt").read_text() == "a"


async def test_validate_and_create_run_with_init(client, session, done_run):
    await seed_dataset(session)
    body = base_spec(name="child")
    body["init_from"] = {"run": "r-nope"}
    r = await client.post("/api/recipes/validate", json={"spec": body})
    assert r.json()["errors"] == ["init_from: run r-nope not found"]
    body["init_from"] = {"run": done_run.id}
    assert (await client.post("/api/recipes", json={"spec": body})).status_code == 201
    r = await client.post("/api/runs", json={"recipe": "child"})
    assert r.status_code == 201, r.text
    detail = (await client.get(f"/api/runs/{r.json()['id']}")).json()
    assert detail["parent_run_id"] == done_run.id and detail["parent_checkpoint"] == "final"
    plain = (await client.get(f"/api/runs/{done_run.id}")).json()
    assert plain["parent_run_id"] is None and plain["parent_checkpoint"] is None


async def test_create_run_sets_parent_and_ands_commercial_ok(client, session, done_run):
    done_run.commercial_ok = False
    await session.commit()
    await seed_dataset(session)
    body = {**base_spec(name="child"), "init_from": {"run": done_run.id}}
    assert (await client.post("/api/recipes", json={"spec": body})).status_code == 201
    rid = (await client.post("/api/runs", json={"recipe": "child"})).json()["id"]
    run = (await session.execute(select(Run).where(Run.id == rid).execution_options(populate_existing=True))).scalar_one()
    assert run.commercial_ok is False and run.parent_run_id == done_run.id


async def test_create_run_422_when_source_checkpoint_gone(client, session, done_run):
    await seed_dataset(session)
    body = {**base_spec(name="child"), "init_from": {"run": done_run.id}}
    assert (await client.post("/api/recipes", json={"spec": body})).status_code == 201
    (RunDir(done_run.run_dir).checkpoints_dir / "final" / ".complete").unlink()
    r = await client.post("/api/runs", json={"recipe": "child"})
    assert r.status_code == 422
    assert r.json()["errors"] == [f"init_from: run {done_run.id} has no complete checkpoint 'final'"]
