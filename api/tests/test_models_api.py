from sqlalchemy import select

from mlops_api.models import ModelVersion
from runs_helpers import finish_files, make_run, set_state


async def done_run(client, session, session_factory, name="r1", metrics=None):
    rid = await make_run(client, session, name=name)
    finish_files((await client.get(f"/api/runs/{rid}")).json()["run_dir"], metrics)
    await set_state(session_factory, rid, state="done")
    return rid


async def test_promote_requires_done(client, session):
    rid = await make_run(client, session)
    assert (await client.post(f"/api/runs/{rid}/promote")).status_code == 409
    assert (await client.post("/api/runs/r-nope/promote")).status_code == 404


async def test_promote_twice_returns_same_version(client, session, session_factory):
    rid = await done_run(client, session, session_factory)
    a = await client.post(f"/api/runs/{rid}/promote")
    b = await client.post(f"/api/runs/{rid}/promote")
    assert a.status_code == 201 and b.status_code == 200 and a.json()["id"] == b.json()["id"]
    body = a.json()
    assert (body["name"], body["number"], body["run_id"], body["task"], body["base_model"]) == (
        "r1", 1, rid, "llm-sft", "m")
    assert body["metrics"] == {"wer": 0.2} and body["commercial_ok"] is True and body["champion"] is False
    assert body["exports"][0]["path"] == "model.bin" and "created_at" in body
    assert len((await session.execute(select(ModelVersion))).scalars().all()) == 1


async def test_promote_numbers_increment_per_name(client, session, session_factory):
    a = await done_run(client, session, session_factory)
    b = await done_run(client, session, session_factory)
    c = await done_run(client, session, session_factory, name="other")
    nums = [(await client.post(f"/api/runs/{r}/promote")).json()["number"] for r in (a, b, c)]
    assert nums == [1, 2, 1]


async def test_list_get_and_filter(client, session, session_factory):
    rid = await done_run(client, session, session_factory)
    mid = (await client.post(f"/api/runs/{rid}/promote")).json()["id"]
    assert [m["id"] for m in (await client.get("/api/models")).json()] == [mid]
    assert (await client.get("/api/models?task=asr")).json() == []
    assert len((await client.get("/api/models?task=llm-sft")).json()) == 1
    assert (await client.get(f"/api/models/{mid}")).json()["id"] == mid
    assert (await client.get("/api/models/999")).status_code == 404


async def test_one_champion_per_task(client, session, session_factory):
    a = await done_run(client, session, session_factory)
    b = await done_run(client, session, session_factory)
    ma = (await client.post(f"/api/runs/{a}/promote")).json()["id"]
    mb = (await client.post(f"/api/runs/{b}/promote")).json()["id"]
    assert (await client.post(f"/api/models/{ma}/champion")).json()["champion"] is True
    assert (await client.post(f"/api/models/{mb}/champion")).json()["champion"] is True
    assert (await client.get(f"/api/models/{ma}")).json()["champion"] is False
    assert (await client.get(f"/api/models/{mb}")).json()["champion"] is True
    assert (await client.post("/api/models/999/champion")).status_code == 404


async def test_export_download(client, session, session_factory):
    rid = await done_run(client, session, session_factory)
    mid = (await client.post(f"/api/runs/{rid}/promote")).json()["id"]
    r = await client.get(f"/api/models/{mid}/exports/0")
    assert r.status_code == 200 and r.content == b"weights"
    assert (await client.get(f"/api/models/{mid}/exports/5")).status_code == 404
    assert (await client.get("/api/models/999/exports/0")).status_code == 404
