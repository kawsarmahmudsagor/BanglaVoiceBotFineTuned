import base64
import binascii
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from .db import get_session
from .models import Upload
from .settings import Settings
from .system import GB, disk_status

router = APIRouter(prefix="/api/uploads")

TUS_VERSION = "1.0.0"
TUS_CONTENT_TYPE = "application/offset+octet-stream"
DEFAULT_FILENAME = "upload.bin"


def _tus(status: int, **headers: str) -> Response:
    return Response(status_code=status, headers={"Tus-Resumable": TUS_VERSION, **headers})


def parse_metadata(header: str | None) -> dict[str, str]:
    """Parse `key base64value,key2 base64value2`. Malformed pairs are skipped."""
    out: dict[str, str] = {}
    for pair in (header or "").split(","):
        parts = pair.strip().split(" ", 1)
        if not parts[0]:
            continue
        value = ""
        if len(parts) == 2:
            try:
                value = base64.b64decode(parts[1].strip(), validate=True).decode("utf-8")
            except (binascii.Error, UnicodeDecodeError):
                continue
        out[parts[0]] = value
    return out


def sanitize_filename(name: str | None) -> str:
    base = (name or "").replace("\\", "/").rsplit("/", 1)[-1].replace("\x00", "").strip()
    return DEFAULT_FILENAME if base in ("", ".", "..") else base


def _path(settings: Settings, upload: Upload) -> Path:
    return settings.incoming_dir / upload.id / upload.filename


def _offset(settings: Settings, upload: Upload) -> int:
    path = _path(settings, upload)
    return path.stat().st_size if path.exists() else 0


@router.options("")
async def options() -> Response:
    return _tus(204, **{"Tus-Version": TUS_VERSION, "Tus-Extension": "creation"})


@router.post("")
async def create(request: Request, session: AsyncSession = Depends(get_session)) -> Response:
    settings: Settings = request.app.state.settings
    try:
        length = int(request.headers["upload-length"])
        if length < 0:
            raise ValueError
    except (KeyError, ValueError):
        return _tus(400)
    disk = disk_status(settings)
    if disk.block or length > (disk.free_gb - settings.disk_min_gb) * GB:
        return _tus(507)
    meta = parse_metadata(request.headers.get("upload-metadata"))
    upload = Upload(
        id=str(uuid.uuid4()),
        filename=sanitize_filename(meta.get("filename")),
        length=length,
        state="complete" if length == 0 else "uploading",
    )
    path = _path(settings, upload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    session.add(upload)
    await session.commit()
    return _tus(201, Location=f"/api/uploads/{upload.id}")


@router.head("/{upload_id}")
async def head(
    upload_id: str, request: Request, session: AsyncSession = Depends(get_session)
) -> Response:
    upload = await session.get(Upload, upload_id)
    if upload is None:
        return _tus(404, **{"Cache-Control": "no-store"})
    offset = _offset(request.app.state.settings, upload)
    return _tus(
        200,
        **{
            "Upload-Offset": str(offset),
            "Upload-Length": str(upload.length),
            "Cache-Control": "no-store",
        },
    )


@router.patch("/{upload_id}")
async def patch(
    upload_id: str, request: Request, session: AsyncSession = Depends(get_session)
) -> Response:
    settings: Settings = request.app.state.settings
    upload = await session.get(Upload, upload_id)
    if upload is None:
        return _tus(404)
    if request.headers.get("content-type", "").split(";")[0].strip().lower() != TUS_CONTENT_TYPE:
        return _tus(415)
    try:
        client_offset = int(request.headers["upload-offset"])
    except (KeyError, ValueError):
        return _tus(400)
    if upload.state in ("complete", "consumed"):
        return _tus(409)  # nothing more may be appended
    path = _path(settings, upload)
    offset = _offset(settings, upload)
    if client_offset != offset:
        return _tus(409, **{"Upload-Offset": str(offset)})
    remaining = upload.length - offset
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > remaining:
        return _tus(413)
    too_big = False
    with open(path, "ab") as f:
        async for chunk in request.stream():
            if len(chunk) > remaining:
                f.write(chunk[:remaining])
                remaining = 0
                too_big = True
                break
            f.write(chunk)
            remaining -= len(chunk)
    new_offset = _offset(settings, upload)
    if new_offset >= upload.length:
        upload.state = "complete"
        await session.commit()
    if too_big:
        return _tus(413, **{"Upload-Offset": str(new_offset)})
    return _tus(204, **{"Upload-Offset": str(new_offset)})
