import base64
from types import SimpleNamespace

import httpx

from mlops_api.app import create_app
from mlops_api.models import Upload
from mlops_api.uploads import parse_metadata, sanitize_filename

TUS = {"Tus-Resumable": "1.0.0"}


def b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


async def create(client, length=10, filename="a.bin"):
    headers = {**TUS, "Upload-Length": str(length)}
    if filename is not None:
        headers["Upload-Metadata"] = f"filename {b64(filename)}"
    return await client.post("/api/uploads", headers=headers)


async def patch(client, loc, offset, body, content_type="application/offset+octet-stream"):
    return await client.patch(
        loc,
        content=body,
        headers={**TUS, "Upload-Offset": str(offset), "Content-Type": content_type},
    )


async def upload_row(session_factory, loc):
    async with session_factory() as s:
        return await s.get(Upload, loc.rsplit("/", 1)[-1])


def client_for(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def usage(free_gb):
    gb = 1024**3
    return SimpleNamespace(total=1000 * gb, used=(1000 - free_gb) * gb, free=free_gb * gb)


async def test_options_advertises_tus(client):
    r = await client.options("/api/uploads")
    assert r.status_code == 204
    assert r.headers["tus-version"] == "1.0.0"
    assert r.headers["tus-extension"] == "creation"
    assert r.headers["tus-resumable"] == "1.0.0"


async def test_upload_in_two_chunks_completes(client, session_factory, settings):
    r = await create(client, length=10, filename="ডেটা.jsonl")
    assert r.status_code == 201 and r.headers["tus-resumable"] == "1.0.0"
    loc = r.headers["location"]
    assert (await patch(client, loc, offset=0, body=b"01234")).headers["upload-offset"] == "5"
    assert (await upload_row(session_factory, loc)).state == "uploading"
    assert (await patch(client, loc, offset=5, body=b"56789")).status_code == 204
    assert (await upload_row(session_factory, loc)).state == "complete"
    f = settings.incoming_dir / loc.rsplit("/", 1)[-1] / "ডেটা.jsonl"
    assert f.read_bytes() == b"0123456789"


async def test_head_reports_offset_and_length(client):
    loc = (await create(client, length=10)).headers["location"]
    await patch(client, loc, offset=0, body=b"abc")
    r = await client.head(loc)
    assert r.status_code == 200
    assert r.headers["upload-offset"] == "3"
    assert r.headers["upload-length"] == "10"
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["tus-resumable"] == "1.0.0"


async def test_wrong_offset_is_409(client):
    loc = (await create(client, length=10)).headers["location"]
    await patch(client, loc, offset=0, body=b"0123")
    assert (await patch(client, loc, offset=2, body=b"xx")).status_code == 409
    assert (await client.head(loc)).headers["upload-offset"] == "4"


async def test_wrong_content_type_is_415(client):
    loc = (await create(client, length=10)).headers["location"]
    r = await patch(client, loc, 0, b"01", content_type="application/json")
    assert r.status_code == 415


async def test_unknown_id_is_404(client):
    assert (await client.head("/api/uploads/nope")).status_code == 404
    assert (await patch(client, "/api/uploads/nope", 0, b"x")).status_code == 404


async def test_patch_on_completed_upload_is_409(client):
    loc = (await create(client, length=3)).headers["location"]
    assert (await patch(client, loc, 0, b"abc")).status_code == 204
    assert (await patch(client, loc, 3, b"d")).status_code == 409


async def test_patch_exceeding_length_is_413_and_file_not_overrun(client, settings):
    loc = (await create(client, length=4)).headers["location"]
    assert (await patch(client, loc, 0, b"0123456")).status_code == 413
    f = settings.incoming_dir / loc.rsplit("/", 1)[-1] / "a.bin"
    assert f.stat().st_size == 0


async def test_missing_upload_length_is_400(client):
    assert (await client.post("/api/uploads", headers=TUS)).status_code == 400


async def test_head_offset_survives_api_restart(settings, session_factory):
    app1, app2 = create_app(settings), create_app(settings)
    async with client_for(app1) as c1:
        loc = (await create(c1, length=10)).headers["location"]
        await patch(c1, loc, offset=0, body=b"0123")
    await app1.state.engine.dispose()
    async with client_for(app2) as c2:
        assert (await c2.head(loc)).headers["upload-offset"] == "4"
        assert (await patch(c2, loc, offset=4, body=b"456789")).status_code == 204
        assert (await upload_row(session_factory, loc)).state == "complete"
    await app2.state.engine.dispose()


async def test_create_refused_when_disk_low(client, monkeypatch):
    monkeypatch.setattr("mlops_api.system.shutil.disk_usage", lambda p: usage(free_gb=5))
    assert (await create(client, length=10)).status_code == 507


async def test_create_refused_when_length_exceeds_headroom(client, monkeypatch):
    monkeypatch.setattr("mlops_api.system.shutil.disk_usage", lambda p: usage(free_gb=12))
    assert (await create(client, length=3 * 1024**3)).status_code == 507
    assert (await create(client, length=1024**3)).status_code == 201


async def test_filename_is_sanitized_to_basename(client, settings):
    r = await create(client, length=1, filename="..\\..\\evil/../x.csv")
    loc = r.headers["location"]
    assert (await patch(client, loc, 0, b"a")).status_code == 204
    assert (settings.incoming_dir / loc.rsplit("/", 1)[-1] / "x.csv").exists()


async def test_missing_filename_defaults(client, settings):
    loc = (await create(client, length=1, filename=None)).headers["location"]
    assert (settings.incoming_dir / loc.rsplit("/", 1)[-1] / "upload.bin").exists()


def test_parse_metadata_and_sanitize():
    header = f"filetype {b64('x')},filename {b64('f.csv')},flag"
    assert parse_metadata(header) == {"filetype": "x", "filename": "f.csv", "flag": ""}
    assert sanitize_filename("a/b\\c.txt") == "c.txt"
    assert sanitize_filename("..") == "upload.bin"
    assert sanitize_filename(None) == "upload.bin"


async def test_cors_exposes_tus_headers(client):
    r = await client.options(
        "/api/uploads",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "PATCH",
            "Access-Control-Request-Headers": "upload-offset,tus-resumable,content-type",
        },
    )
    assert r.status_code == 200
    r = await client.head("/api/uploads/nope", headers={"Origin": "http://localhost:3000"})
    exposed = r.headers["access-control-expose-headers"]
    assert "Upload-Offset" in exposed and "Location" in exposed
