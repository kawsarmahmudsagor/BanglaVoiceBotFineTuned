"""End to end: tus upload -> ingest -> recipe -> run -> train/eval/export -> promote -> champion.

Drives the real ASGI app and the worker's tick functions with the fake backend. App and worker
share one sqlite file and one data root through the `settings` fixture.
"""
import json

import pytest

from workers import executor
from workers.__main__ import gpu_tick
from workers.executor import NullForwarder
from workers.ingest_job import cpu_tick

from test_uploads import TUS, b64

TURNS = [
    ("আপনার নাম কী?", "আমার নাম বাংলা ভয়েস সহকারী।"),
    ("আজকের আবহাওয়া কেমন?", "আজ আকাশ পরিষ্কার এবং তাপমাত্রা মনোরম।"),
    ("ঢাকা থেকে চট্টগ্রাম কত দূর?", "ঢাকা থেকে চট্টগ্রাম প্রায় আড়াইশো কিলোমিটার দূরে।"),
]


def jsonl_bytes() -> bytes:
    rows = [
        {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}
        for q, a in TURNS
    ]
    return ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8")


async def tus_upload(client, data: bytes, filename: str) -> str:
    r = await client.post(
        "/api/uploads",
        headers={**TUS, "Upload-Length": str(len(data)), "Upload-Metadata": f"filename {b64(filename)}"},
    )
    assert r.status_code == 201
    loc = r.headers["location"]
    r = await client.patch(
        loc, content=data,
        headers={**TUS, "Upload-Offset": "0", "Content-Type": "application/offset+octet-stream"},
    )
    assert r.status_code == 204
    return loc.rsplit("/", 1)[-1]


@pytest.mark.parametrize("dataset_commercial", [True, False])
async def test_pipeline_end_to_end(client, settings, session_factory, monkeypatch, dataset_commercial):
    monkeypatch.setattr(executor, "POLL_S", 0.05)

    # upload + ingest
    upload_id = await tus_upload(client, jsonl_bytes(), "bangla.jsonl")
    r = await client.post("/api/datasets/ingest", json={
        "upload_id": upload_id, "name": "bangla-chat", "task": "llm-sft",
        "provenance": {"source": "test", "license": "cc-by", "consent": "yes",
                       "commercial_ok": dataset_commercial},
    })
    assert r.status_code == 202
    ingest_id = r.json()["ingest_id"]
    assert await cpu_tick(session_factory, settings) is True
    ing = (await client.get(f"/api/ingests/{ingest_id}")).json()
    assert ing["state"] == "done", ing
    assert ing["version"]["number"] == 1 and ing["version"]["n_samples"] == 3
    assert ing["version"]["commercial_ok"] is dataset_commercial

    # recipe + run
    r = await client.post("/api/recipes", json={"spec": {
        "name": "e2e-recipe", "task": "llm-sft", "backend": "fake", "base_model": "fake-base",
        "base_model_meta": {"license": "mit", "commercial_ok": True},
        "datasets": {"train": "bangla-chat@v1"}, "method": {"type": "none"},
        "hparams": {"steps": 3}, "export": ["dummy"],
    }})
    assert r.status_code == 201, r.text
    r = await client.post("/api/runs", json={"recipe": "e2e-recipe"})
    assert r.status_code == 201, r.text
    run_id = r.json()["id"]
    assert (await client.get(f"/api/runs/{run_id}")).json()["commercial_ok"] is dataset_commercial

    # train / eval / export
    assert await gpu_tick(session_factory, settings, lambda s: NullForwarder()) is True
    run = (await client.get(f"/api/runs/{run_id}")).json()
    assert run["state"] == "done", run
    metrics = (await client.get(f"/api/runs/{run_id}/metrics")).json()["items"]
    assert len([m for m in metrics if (m["split"], m["name"]) == ("train", "loss")]) == 3
    assert "fake_score" in run["eval"]["metrics"]
    assert [e["format"] for e in run["exports"]] == ["dummy"]

    # promote + champion
    r = await client.post(f"/api/runs/{run_id}/promote")
    assert r.status_code == 201
    model = r.json()
    assert (await client.get(f"/api/models/{model['id']}/exports/0")).status_code == 200
    r = await client.post(f"/api/models/{model['id']}/champion")
    assert r.status_code == 200 and r.json()["champion"] is True

    listed = (await client.get("/api/models", params={"task": "llm-sft"})).json()
    assert [(m["id"], m["champion"], m["commercial_ok"]) for m in listed] == [
        (model["id"], True, dataset_commercial)
    ]
