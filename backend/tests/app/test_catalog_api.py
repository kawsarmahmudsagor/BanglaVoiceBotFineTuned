"""Read-only catalog endpoints."""
from __future__ import annotations

import copy

import yaml


def valid_entry(id: str = "whisper-x", task: str = "asr") -> dict:
    e = {
        "id": id,
        "task": task,
        "name": f"Model {id}",
        "hf_repo": f"org/{id}",
        "params": "809M",
        "role": "primary",
        "license": {"id": "mit", "commercial_ok": True, "url": "https://example.com/l"},
        "bangla": {"status": "native", "evidence": "card lists bn"},
        "runtime": {"engine": "faster_whisper", "streaming": True},
        "description": "A model.",
        "dataset": {"format": task, "required_fields": ["file", "text"], "example": "a.wav,আমি",
                    "recommended": {"hours": "10-100"}},
        "finetune": {"methods": ["full"], "frameworks": ["transformers"], "targets": ["decoder"],
                     "hyperparams": {"lr": 1e-5}, "metrics": ["wer"], "pitfalls": ["forgetting"]},
        "recipe_template": {"method": {"type": "full"}, "hparams": {"lr": 1e-5}},
        "sources": [{"title": "card", "url": "https://example.com"}],
        "verified": "2026-11-23",
    }
    return copy.deepcopy(e)


def write(settings, section: str, stem: str, data) -> None:
    d = settings.catalog_dir / section
    d.mkdir(parents=True, exist_ok=True)
    text = data if isinstance(data, str) else yaml.safe_dump(data, allow_unicode=True)
    (d / f"{stem}.yaml").write_text(text, encoding="utf-8")


async def test_list_all_and_errors(client, settings):
    write(settings, "stt", "whisper-x", valid_entry())
    write(settings, "stt", "broken", "id: [x\n")
    body = (await client.get("/api/catalog")).json()
    assert [e["id"] for e in body["entries"]] == ["whisper-x"]
    assert [e["file"] for e in body["errors"]] == ["stt/broken.yaml"]
    assert body["errors"][0]["message"]


async def test_list_filters_by_task(client, settings):
    write(settings, "stt", "whisper-x", valid_entry())
    write(settings, "tts", "voice", valid_entry("voice", "tts"))
    body = (await client.get("/api/catalog", params={"task": "tts"})).json()
    assert [e["id"] for e in body["entries"]] == ["voice"]


async def test_invalid_task_422(client):
    assert (await client.get("/api/catalog", params={"task": "nope"})).status_code == 422


async def test_detail_and_404(client, settings):
    write(settings, "stt", "whisper-x", valid_entry())
    r = await client.get("/api/catalog/whisper-x")
    assert r.status_code == 200
    body = r.json()
    assert body["dataset"]["example"] == "a.wav,আমি"
    assert body["verified"] == "2026-11-23"
    r = await client.get("/api/catalog/nope")
    assert r.status_code == 404
    assert r.json()["detail"] == "catalog entry 'nope' not found"


async def test_summary_has_documented_fields(client, settings):
    write(settings, "stt", "whisper-x", valid_entry())
    (s,) = (await client.get("/api/catalog")).json()["entries"]
    assert set(s) == {"id", "task", "name", "hf_repo", "params", "role", "license", "bangla_status", "streaming"}
