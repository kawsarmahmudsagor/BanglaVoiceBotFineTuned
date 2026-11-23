import base64
import io
import json
import zipfile

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from mlops_api.models import Ingest
from mlops_api.datasets.versions import run_ingest

TUS = {"Tus-Resumable": "1.0.0"}


def wav(seconds=1.0, freq=220.0, sr=16000) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    buf = io.BytesIO()
    sf.write(buf, (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def jsonl_bytes(n, offset=0) -> bytes:
    rows = [
        {"messages": [{"role": "user", "content": f"প্রশ্ন {i + offset}"},
                      {"role": "assistant", "content": f"উত্তর {i + offset}"}]}
        for i in range(n)
    ]
    return ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8")


def zip_bytes(items: list[tuple[str, str, bytes]]) -> bytes:
    """items: (relative file, transcript, audio bytes)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        lines = ["file,text"] + [f"{f},{t}" for f, t, _ in items]
        zf.writestr("metadata.csv", "\n".join(lines).encode("utf-8"))
        for f, _, data in items:
            zf.writestr(f, data)
    return buf.getvalue()


async def upload_bytes(client, filename, data) -> str:
    meta = f"filename {base64.b64encode(filename.encode()).decode()}"
    r = await client.post("/api/uploads", headers={**TUS, "Upload-Length": str(len(data)),
                                                  "Upload-Metadata": meta})
    loc = r.headers["location"]
    await client.patch(loc, content=data, headers={**TUS, "Upload-Offset": "0",
                                                   "Content-Type": "application/offset+octet-stream"})
    return loc.rsplit("/", 1)[-1]


def req(upload_id, name, task="llm-sft", eval_only=False, commercial_ok=True):
    return {"upload_id": upload_id, "name": name, "task": task, "eval_only": eval_only,
            "provenance": {"source": "test", "license": "cc-by", "consent": "yes",
                           "commercial_ok": commercial_ok, "notes": ""}}


async def run_ingest_now(session_factory, iid, settings):
    async with session_factory() as s:
        await run_ingest(s, iid, settings)


async def ingest_via_api(client, session_factory, settings, filename, data, name, task="llm-sft", **kw):
    up = await upload_bytes(client, filename, data)
    r = await client.post("/api/datasets/ingest", json=req(up, name, task, **kw))
    assert r.status_code == 202, r.text
    iid = r.json()["ingest_id"]
    await run_ingest_now(session_factory, iid, settings)
    return iid


async def test_ingest_flow_creates_v1_then_v2(client, session_factory, settings):
    iid = await ingest_via_api(client, session_factory, settings, "a.jsonl", jsonl_bytes(2), "ecom-instruct")
    body = (await client.get(f"/api/ingests/{iid}")).json()
    assert body["state"] == "done" and body["version"]["number"] == 1 and body["error"] is None
    await ingest_via_api(client, session_factory, settings, "b.jsonl", jsonl_bytes(3, offset=10), "ecom-instruct")
    d = (await client.get("/api/datasets/ecom-instruct")).json()
    assert [v["number"] for v in d["versions"]] == [1, 2]
    listing = (await client.get("/api/datasets")).json()
    assert listing[0]["name"] == "ecom-instruct" and listing[0]["task"] == "llm-sft"
    latest = listing[0]["latest"]
    assert latest["number"] == 2 and latest["n_samples"] == 3 and latest["commercial_ok"] is True
    assert latest["created_at"].endswith("+00:00")


async def test_version_detail_and_samples(client, session_factory, settings):
    await ingest_via_api(client, session_factory, settings, "a.jsonl", jsonl_bytes(5), "ecom-instruct")
    v = (await client.get("/api/datasets/ecom-instruct/versions/1")).json()
    assert v["n_samples"] == 5 and len(v["hash"]) == 16
    assert sum(v["splits"].values()) == 5
    assert v["provenance"]["license"] == "cc-by" and v["parent_number"] is None
    assert "errors" in v["report"]
    page = (await client.get("/api/datasets/ecom-instruct/versions/1/samples?limit=2")).json()
    assert page["total"] == 5 and len(page["rows"]) == 2 and "messages_json" in page["rows"][0]
    page = (await client.get("/api/datasets/ecom-instruct/versions/1/samples?offset=4&limit=2")).json()
    assert len(page["rows"]) == 1
    assert (await client.get("/api/datasets/ecom-instruct/versions/9")).status_code == 404
    assert (await client.get("/api/datasets/nope")).status_code == 404


async def test_name_reused_with_other_task_is_409(client, session_factory, settings):
    await ingest_via_api(client, session_factory, settings, "a.jsonl", jsonl_bytes(1), "mixed")
    up = await upload_bytes(client, "z.zip", b"zip")
    r = await client.post("/api/datasets/ingest", json=req(up, "mixed", task="asr"))
    assert r.status_code == 409
    assert r.json()["detail"] == "dataset 'mixed' is task 'llm-sft'"


async def test_ingest_request_validation(client):
    up = await upload_bytes(client, "a.jsonl", jsonl_bytes(1))
    assert (await client.post("/api/datasets/ingest", json=req(up, "Bad Name"))).status_code == 422
    assert (await client.post("/api/datasets/ingest", json=req(up, "ok", task="nlp"))).status_code == 422
    assert (await client.post("/api/datasets/ingest", json=req("nope", "ok"))).status_code == 404
    # incomplete upload
    r = await client.post("/api/uploads", headers={**TUS, "Upload-Length": "10",
                                                  "Upload-Metadata": "filename " + base64.b64encode(b"x.jsonl").decode()})
    partial = r.headers["location"].rsplit("/", 1)[-1]
    assert (await client.post("/api/datasets/ingest", json=req(partial, "ok"))).status_code == 409


async def test_failed_validation_keeps_report_and_creates_no_version(client, session_factory, settings):
    iid = await ingest_via_api(client, session_factory, settings, "a.jsonl", b"not json\n", "broken")
    body = (await client.get(f"/api/ingests/{iid}")).json()
    assert body["state"] == "failed" and body["version"] is None
    assert body["report"]["errors"][0]["code"] == "invalid_json"
    assert (await client.get("/api/datasets/broken")).status_code == 404  # no dataset, no version


async def test_run_ingest_never_raises_on_exception(client, session_factory, settings):
    iid = await ingest_via_api(client, session_factory, settings, "a.zip", b"PK\x03\x04garbage", "badzip", task="asr")
    body = (await client.get(f"/api/ingests/{iid}")).json()
    assert body["state"] == "failed"
    assert body["error"] or body["report"]["errors"]


async def test_run_ingest_exception_sets_error(client, session_factory, settings, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("mlops_api.datasets.versions.ingest", boom)
    iid = await ingest_via_api(client, session_factory, settings, "a.jsonl", jsonl_bytes(1), "boomy")
    body = (await client.get(f"/api/ingests/{iid}")).json()
    assert body["state"] == "failed" and body["error"] == "kaboom"


async def asr_dataset(client, session_factory, settings, name="stt-a"):
    data = zip_bytes([("a.wav", "এক", wav(1.0, 220)), ("b.wav", "দুই", wav(1.0, 330)),
                      ("c.wav", "তিন", wav(0.2, 440))])
    await ingest_via_api(client, session_factory, settings, "d.zip", data, name, task="asr")


async def test_samples_filter_by_warning_and_split(client, session_factory, settings):
    await asr_dataset(client, session_factory, settings)
    v = (await client.get("/api/datasets/stt-a/versions/1")).json()
    assert v["n_samples"] == 3
    warned = [w for w in v["report"]["warnings"] if w["code"] == "duration_out_of_range"]
    assert len(warned) == 1
    r = (await client.get("/api/datasets/stt-a/versions/1/samples?warning=duration_out_of_range")).json()
    assert r["total"] == 1 and r["rows"][0]["sample_id"] == warned[0]["sample_id"]
    assert (await client.get("/api/datasets/stt-a/versions/1/samples?warning=clipping")).json()["total"] == 0
    all_rows = (await client.get("/api/datasets/stt-a/versions/1/samples")).json()["rows"]
    split = all_rows[0]["split"]
    r = (await client.get(f"/api/datasets/stt-a/versions/1/samples?split={split}")).json()
    assert r["total"] == sum(1 for x in all_rows if x["split"] == split)


async def test_audio_served_and_404_for_llm(client, session_factory, settings):
    await asr_dataset(client, session_factory, settings)
    rows = (await client.get("/api/datasets/stt-a/versions/1/samples")).json()["rows"]
    r = await client.get(f"/api/datasets/stt-a/versions/1/audio/{rows[0]['sample_id']}")
    assert r.status_code == 200 and r.content[:4] == b"RIFF"
    assert (await client.get("/api/datasets/stt-a/versions/1/audio/nope")).status_code == 404
    await ingest_via_api(client, session_factory, settings, "a.jsonl", jsonl_bytes(1), "llm-a")
    sid = (await client.get("/api/datasets/llm-a/versions/1/samples")).json()["rows"][0]["sample_id"]
    assert (await client.get(f"/api/datasets/llm-a/versions/1/audio/{sid}")).status_code == 404


async def test_edit_transcript_creates_child_version(client, session_factory, settings):
    await asr_dataset(client, session_factory, settings)
    rows1 = (await client.get("/api/datasets/stt-a/versions/1/samples")).json()["rows"]
    target = rows1[0]
    r = await client.post("/api/datasets/stt-a/versions/1/edits",
                          json={"sample_id": target["sample_id"], "text": "নতুন"})
    assert r.status_code == 201 and r.json()["number"] == 2
    v2 = (await client.get("/api/datasets/stt-a/versions/2")).json()
    assert v2["parent_number"] == 1 and v2["hash"] != (await client.get("/api/datasets/stt-a/versions/1")).json()["hash"]
    rows2 = (await client.get("/api/datasets/stt-a/versions/2/samples")).json()["rows"]
    assert "নতুন" in [x["text"] for x in rows2]
    assert target["text"] not in [x["text"] for x in rows2]
    old = (await client.get("/api/datasets/stt-a/versions/1/samples")).json()["rows"]
    assert "নতুন" not in [x["text"] for x in old]
    edited = next(x for x in rows2 if x["text"] == "নতুন")
    assert edited["sample_id"] != target["sample_id"] and edited["split"] == target["split"]
    assert edited["file"] == target["file"]
    # audio is reachable from the new version
    a = await client.get(f"/api/datasets/stt-a/versions/2/audio/{edited['sample_id']}")
    assert a.status_code == 200
    # manifest on disk is valid
    path = settings.datasets_dir / "stt-a" / v2["hash"] / "manifest.parquet"
    assert pq.read_table(path).num_rows == 3


async def test_edit_errors(client, session_factory, settings):
    await asr_dataset(client, session_factory, settings)
    r = await client.post("/api/datasets/stt-a/versions/1/edits", json={"sample_id": "zzz", "text": "x"})
    assert r.status_code == 404
    await ingest_via_api(client, session_factory, settings, "a.jsonl", jsonl_bytes(1), "llm-a")
    sid = (await client.get("/api/datasets/llm-a/versions/1/samples")).json()["rows"][0]["sample_id"]
    r = await client.post("/api/datasets/llm-a/versions/1/edits", json={"sample_id": sid, "text": "x"})
    assert r.status_code == 400


async def test_patch_eval_only(client, session_factory, settings):
    await ingest_via_api(client, session_factory, settings, "a.jsonl", jsonl_bytes(1), "ecom-instruct")
    r = await client.patch("/api/datasets/ecom-instruct", json={"eval_only": True})
    assert r.status_code == 200 and r.json()["eval_only"] is True
    assert (await client.get("/api/datasets")).json()[0]["eval_only"] is True
    assert (await client.patch("/api/datasets/nope", json={"eval_only": True})).status_code == 404


async def test_ingest_unknown_id_is_404(client):
    assert (await client.get("/api/ingests/999")).status_code == 404


async def test_edit_remaps_report_sample_ids(client, session_factory, settings):
    await asr_dataset(client, session_factory, settings)
    v1 = (await client.get("/api/datasets/stt-a/versions/1")).json()
    old = next(w["sample_id"] for w in v1["report"]["warnings"] if w["code"] == "duration_out_of_range")
    r = await client.post("/api/datasets/stt-a/versions/1/edits", json={"sample_id": old, "text": "চার"})
    assert r.status_code == 201
    got = (await client.get("/api/datasets/stt-a/versions/2/samples?warning=duration_out_of_range")).json()
    assert got["total"] == 1 and got["rows"][0]["text"] == "চার"
    v2 = (await client.get("/api/datasets/stt-a/versions/2")).json()
    assert old not in [w["sample_id"] for w in v2["report"]["warnings"]]
    # parent report untouched
    assert old in [w["sample_id"] for w in (await client.get("/api/datasets/stt-a/versions/1")).json()["report"]["warnings"]]


async def test_run_ingest_falls_back_to_fresh_session_when_fail_breaks(client, session_factory, settings, monkeypatch):
    import mlops_api.datasets.versions as vs

    def boom(*a, **k):
        raise RuntimeError("kaboom")

    real_fail = vs._fail
    calls = {"n": 0}

    async def flaky_fail(session, ingest_id, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("db hiccup")
        return await real_fail(session, ingest_id, **kw)

    monkeypatch.setattr(vs, "ingest", boom)
    monkeypatch.setattr(vs, "_fail", flaky_fail)
    iid = await ingest_via_api(client, session_factory, settings, "a.jsonl", jsonl_bytes(1), "flaky")
    body = (await client.get(f"/api/ingests/{iid}")).json()
    assert body["state"] == "failed" and body["error"] == "kaboom"


async def test_run_ingest_on_done_ingest_is_noop(client, session_factory, settings):
    iid = await ingest_via_api(client, session_factory, settings, "a.jsonl", jsonl_bytes(1), "once")
    await run_ingest_now(session_factory, iid, settings)
    assert [v["number"] for v in (await client.get("/api/datasets/once")).json()["versions"]] == [1]
    assert (await client.get(f"/api/ingests/{iid}")).json()["state"] == "done"


async def test_new_version_applies_request_eval_only_to_existing_dataset(client, session_factory, settings):
    await ingest_via_api(client, session_factory, settings, "a.jsonl", jsonl_bytes(1), "evalset")
    assert (await client.get("/api/datasets/evalset")).json()["eval_only"] is False
    await ingest_via_api(client, session_factory, settings, "b.jsonl", jsonl_bytes(2, offset=5), "evalset",
                         eval_only=True)
    assert (await client.get("/api/datasets/evalset")).json()["eval_only"] is True


async def test_provenance_source_license_consent_required(client):
    up = await upload_bytes(client, "a.jsonl", jsonl_bytes(1))
    for field in ("source", "license", "consent"):
        body = req(up, "prov")
        body["provenance"][field] = ""
        assert (await client.post("/api/datasets/ingest", json=body)).status_code == 422, field
        del body["provenance"][field]
        assert (await client.post("/api/datasets/ingest", json=body)).status_code == 422, field
    assert (await client.post("/api/datasets/ingest", json=req(up, "prov"))).status_code == 202


async def test_upload_dir_deleted_after_done_kept_after_failed(client, session_factory, settings):
    ok_up = await upload_bytes(client, "a.jsonl", jsonl_bytes(1))
    r = await client.post("/api/datasets/ingest", json=req(ok_up, "keep"))
    await run_ingest_now(session_factory, r.json()["ingest_id"], settings)
    assert (await client.get(f"/api/ingests/{r.json()['ingest_id']}")).json()["state"] == "done"
    assert not (settings.incoming_dir / ok_up).exists()

    bad_up = await upload_bytes(client, "b.jsonl", b"not json\n")
    r = await client.post("/api/datasets/ingest", json=req(bad_up, "broken"))
    await run_ingest_now(session_factory, r.json()["ingest_id"], settings)
    assert (await client.get(f"/api/ingests/{r.json()['ingest_id']}")).json()["state"] == "failed"
    assert (settings.incoming_dir / bad_up).is_dir()
