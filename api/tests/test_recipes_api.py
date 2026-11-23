import pytest
import yaml
from sqlalchemy import select

from mlops_api.models import Dataset, DatasetVersion, RecipeVersion

HEADER = "# Generated from the platform database. Edit in the UI."


def spec(name="r1", train="d1@v1", eval_=None, task="llm-sft", steps=5):
    return {
        "name": name, "task": task, "backend": "fake", "base_model": "m",
        "base_model_meta": {"license": "mit", "commercial_ok": True},
        "datasets": {"train": train, **({"eval": eval_} if eval_ else {})},
        "method": {"type": "none"}, "hparams": {"steps": steps}, "export": ["dummy"],
    }


async def seed(session, name="d1", task="llm-sft", eval_only=False, versions=1):
    ds = Dataset(name=name, task=task, eval_only=eval_only)
    session.add(ds)
    await session.flush()
    for n in range(1, versions + 1):
        session.add(DatasetVersion(dataset_id=ds.id, number=n, hash="h", path="p", n_samples=1))
    await session.commit()


async def test_backends(client):
    r = await client.get("/api/backends")
    assert r.status_code == 200
    fake = next(b for b in r.json() if b["name"] == "fake")
    assert "schema" in fake["tasks"]["llm-sft"]


async def test_save_writes_file_and_overwrites_unfrozen(client, session, settings):
    await seed(session)
    r = await client.post("/api/recipes", json={"spec": spec()})
    assert r.status_code == 201, r.text
    assert r.json() == {"name": "r1", "number": 1}
    text = (settings.recipes_dir / "r1.yaml").read_text(encoding="utf-8")
    assert text.startswith(HEADER + "\n")
    assert yaml.safe_load(text)["name"] == "r1"
    r = await client.post("/api/recipes", json={"spec": spec(steps=9)})
    assert r.json()["number"] == 1
    d = await client.get("/api/recipes/r1/versions/1")
    assert d.json()["spec"]["hparams"]["steps"] == 9
    assert "steps: 9" in d.json()["yaml"]


async def test_save_after_freeze_creates_new_version(client, session):
    await seed(session)
    await client.post("/api/recipes", json={"spec": spec()})
    v = (await session.execute(select(RecipeVersion))).scalar_one()
    v.frozen = True
    await session.commit()
    r = await client.post("/api/recipes", json={"spec": spec()})
    assert r.json()["number"] == 2
    detail = (await client.get("/api/recipes/r1")).json()
    assert [x["number"] for x in detail["versions"]] == [1, 2]
    listing = (await client.get("/api/recipes")).json()
    assert listing[0]["name"] == "r1" and listing[0]["latest"]["number"] == 2
    assert listing[0]["latest"]["frozen"] is False


async def test_eval_only_train_dataset(client, session):
    await seed(session, eval_only=True)
    r = await client.post("/api/recipes", json={"spec": spec()})
    assert r.status_code == 422
    assert "dataset 'd1' is eval-only and cannot be used for training" in r.json()["errors"]


async def test_missing_dataset(client, session):
    await seed(session)
    r = await client.post("/api/recipes/validate", json={"spec": spec(train="d1@v2")})
    assert r.status_code == 200
    assert "dataset 'd1@v2' not found" in r.json()["errors"]


async def test_task_mismatch(client, session):
    await seed(session, task="asr")
    r = await client.post("/api/recipes/validate", json={"spec": spec()})
    assert "dataset 'd1' is task 'asr', recipe is task 'llm-sft'" in r.json()["errors"]


async def test_eval_only_ok_for_eval_slot(client, session):
    await seed(session)
    await seed(session, name="e1", eval_only=True)
    r = await client.post("/api/recipes/validate", json={"spec": spec(eval_="e1@v1")})
    assert r.json()["errors"] == []


async def test_yaml_syntax_error(client):
    r = await client.post("/api/recipes/validate", json={"yaml": "a: [unclosed"})
    assert r.status_code == 200
    assert r.json()["errors"][0].startswith("yaml: ")
    assert r.json()["spec"] is None


async def test_yaml_body_and_pydantic_errors(client, session):
    await seed(session)
    r = await client.post("/api/recipes/validate", json={"yaml": yaml.safe_dump(spec())})
    assert r.json()["errors"] == [] and r.json()["spec"]["name"] == "r1"
    bad = spec()
    del bad["base_model"]
    r = await client.post("/api/recipes", json={"spec": bad})
    assert r.status_code == 422
    assert any(e.startswith("base_model: ") for e in r.json()["errors"])


async def test_body_needs_exactly_one(client):
    assert (await client.post("/api/recipes/validate", json={})).status_code == 422
    r = await client.post("/api/recipes/validate", json={"yaml": "a: 1", "spec": {}})
    assert r.status_code == 422


async def test_schema_error_reported(client, session):
    await seed(session)
    r = await client.post("/api/recipes", json={"spec": spec(steps=0)})
    assert r.status_code == 422 and r.json()["errors"]


async def test_clone(client, session, settings):
    await seed(session)
    await client.post("/api/recipes", json={"spec": spec()})
    r = await client.post("/api/recipes/r1/clone", json={"new_name": "r2"})
    assert r.status_code == 201
    d = (await client.get("/api/recipes/r2/versions/1")).json()
    assert d["spec"]["name"] == "r2" and d["spec"]["hparams"] == {"steps": 5}
    assert "name: r2" in d["yaml"]
    assert (settings.recipes_dir / "r2.yaml").exists()
    assert (await client.post("/api/recipes/r1/clone", json={"new_name": "r2"})).status_code == 409
    assert (await client.post("/api/recipes/r1/clone", json={"new_name": "Bad Name"})).status_code == 422
    assert (await client.post("/api/recipes/r1/clone", json={"new_name": "r2\n"})).status_code == 422
    assert (await client.post("/api/recipes/nope/clone", json={"new_name": "r3"})).status_code == 404


async def test_404s(client):
    assert (await client.get("/api/recipes/nope")).status_code == 404
    assert (await client.get("/api/recipes/nope/versions/1")).status_code == 404
