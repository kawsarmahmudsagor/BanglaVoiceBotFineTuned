"""Studio routes (spec 6): add one video by URL or upload, preview media, start with keep ranges, list clips.

Errors carry `detail: {error, ...}` like the rest of the API; duplicates add `video_id` (409).

Upload = the raw file bytes as the request body with `?filename=` (no multipart: python-multipart is not a
dependency). The body is streamed into `<dataset>/.upload-<uuid>/download/upload<ext>` in 1 MB writes, hashed
while it is written, and the temp folder is renamed to the video folder once the id is known.
"""
from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import re
import shutil
import uuid
from pathlib import Path, PureWindowsPath
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.requests import ClientDisconnect

from .. import models
from ..pipeline import STAGES
from ..pipeline.cut import BadCuts, normalize_ranges, to_orig
from ..pipeline.download import expand_urls
from ..runner import cut_dir, video_dir
from .jobs import _video_or_404, dataset_or_404
from .review import QUEUE_STATUSES, review_reason

router = APIRouter(prefix="/api/datasets")

CHUNK = 1 << 20
_SUFFIX = re.compile(r"^\.[a-z0-9]{1,10}$")


class AddUrl(BaseModel):
    url: str


class StartBody(BaseModel):
    cuts: Any = None  # validated by hand so every bad value is a 422 with `detail.error`


def _err(status: int, error: str, **extra: Any) -> HTTPException:
    return HTTPException(status, {"error": error, **extra})


def _exists(store, dataset_id: str, video_id: str) -> bool:
    return any(v["id"] == video_id for v in store.list_videos(dataset_id))


# --- add by URL
@router.post("/{dataset_id}/studio/videos")
async def add_url(dataset_id: str, body: AddUrl, request: Request) -> dict:
    dataset_or_404(request, dataset_id)
    url = body.url.strip()
    if not url:
        raise _err(400, "no URL")
    try:
        pairs = await run_in_threadpool(expand_urls, url)
    except Exception as e:
        raise _err(400, str(e)) from e
    if len(pairs) != 1:
        raise _err(400, "Studio takes one video at a time; this looks like a playlist" if pairs else "no video found")
    vid, vurl = pairs[0]
    store = request.app.state.store
    if not store.add_video(dataset_id, vid, vurl, origin="studio", source="youtube", status="fetching"):
        raise _err(409, "video already in this dataset", video_id=vid)
    request.app.state.runner.enqueue(dataset_id)
    return {"video_id": vid}


# --- upload
def _write_chunk(f, chunk: bytes) -> None:
    f.write(chunk)


class _Sink:
    """Buffered, hashed writer of the uploaded bytes (1 MB writes, off the event loop)."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "wb")
        self.sha1, self.buf, self.size = hashlib.sha1(), bytearray(), 0

    async def add(self, data: bytes) -> None:
        self.sha1.update(data)
        self.size += len(data)
        self.buf += data
        while len(self.buf) >= CHUNK:
            chunk = bytes(self.buf[:CHUNK])
            del self.buf[:CHUNK]
            await run_in_threadpool(_write_chunk, self.f, chunk)

    async def finish(self) -> None:
        if self.buf:
            await run_in_threadpool(_write_chunk, self.f, bytes(self.buf))
            self.buf.clear()


async def _receive(request: Request, target: Path) -> tuple[str, int]:
    """Stream the raw body to `target`, hashing as it goes. Returns (sha1 hex, size). The file is closed on
    every path, so the caller can delete it (Windows keeps open files)."""
    sink = _Sink(target)
    try:
        async for chunk in request.stream():
            if chunk:
                await sink.add(chunk)
        await sink.finish()
    finally:
        sink.f.close()
    return sink.sha1.hexdigest(), sink.size


@router.post("/{dataset_id}/studio/upload")
async def upload(dataset_id: str, request: Request, filename: str | None = None) -> dict:
    """Body = the file's bytes (any Content-Type); `?filename=` = its original name (title, suffix)."""
    dataset_or_404(request, dataset_id)
    name = PureWindowsPath((filename or "").strip()).name.strip()
    if not name:
        raise _err(400, "filename query parameter is required")
    suffix = Path(name).suffix.lower()
    store, data_dir = request.app.state.store, request.app.state.settings.data_dir
    tmp = Path(data_dir) / "videos" / dataset_id / f".upload-{uuid.uuid4().hex}"
    done = False
    try:
        try:
            sha1, size = await _receive(request, tmp / "download" / f"upload{suffix if _SUFFIX.match(suffix) else ''}")
        except ClientDisconnect as e:
            raise _err(400, "upload interrupted") from e
        except Exception as e:
            raise _err(500, f"upload failed: {e}") from e
        if size == 0:
            raise _err(400, "the uploaded file is empty")
        vid = "local_" + sha1[:11]
        # no await from here to add_video: concurrent uploads of the same bytes cannot both pass the check
        if _exists(store, dataset_id, vid):
            raise _err(409, "this file is already in the dataset", video_id=vid)
        dest = video_dir(data_dir, dataset_id, vid)
        if dest.exists():  # leftovers of a deleted video whose folder could not be removed
            shutil.rmtree(dest)
        tmp.rename(dest)
        store.add_video(dataset_id, vid, f"local:{name}", origin="studio", source="local", status="fetching")
        store.update_video(dataset_id, vid, title=name)
        done = True
    finally:
        if not done:
            shutil.rmtree(tmp, ignore_errors=True)
    request.app.state.runner.enqueue(dataset_id)
    return {"video_id": vid}


# --- list, media
def _cuts(raw: str | None) -> list | None:
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return None


def _media_path(data_dir: Path, dataset_id: str, v: dict) -> Path | None:
    if not v.get("media"):
        return None
    base = video_dir(data_dir, dataset_id, v["id"]).resolve()
    path = (base / v["media"]).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        return None
    return path


def _attention(c: dict) -> bool:
    return c["status"] in QUEUE_STATUSES or c.get("flag") is not None


@router.get("/{dataset_id}/studio/videos")
def list_studio_videos(dataset_id: str, request: Request) -> list[dict]:
    store, data_dir = request.app.state.store, request.app.state.settings.data_dir
    dataset_or_404(request, dataset_id)
    clips = store.clips(dataset_id)
    out = []
    for v in store.list_videos(dataset_id):
        if v.get("origin") != "studio":
            continue
        mine = [c for c in clips if c["video_id"] == v["id"]]
        dropped = [c for c in mine if c["status"] in ("dropped", "rejected")]
        out.append({
            "id": v["id"], "title": v["title"], "url": v["url"], "source": v["source"], "status": v["status"],
            "error": v["error"], "warning": v["warning"], "duration_s": v["duration_s"],
            "media_ready": _media_path(data_dir, dataset_id, v) is not None,
            "cuts": _cuts(v.get("cuts")),
            "stages": {s: store.stage_done(dataset_id, v["id"], s) for s in STAGES},
            "clips": {
                "total": len(mine),
                "kept": len(mine) - len(dropped),
                "flagged": sum(1 for c in mine if c not in dropped and _attention(c)),
                "dropped": len(dropped),
            },
        })
    return out


@router.get("/{dataset_id}/videos/{video_id}/media")
def video_media(dataset_id: str, video_id: str, request: Request) -> FileResponse:
    v = _video_or_404(request, dataset_id, video_id)
    path = _media_path(request.app.state.settings.data_dir, dataset_id, v)
    if path is None:
        raise HTTPException(404, "preview not available")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type)  # Starlette answers Range requests with 206


# --- start
def _check_cuts(cuts: Any) -> list[list[float]]:
    if cuts is None:
        return []
    if not isinstance(cuts, list):
        raise _err(422, "cuts must be a list of [start_s, end_s] ranges")
    out = []
    for c in cuts:
        if not isinstance(c, list) or len(c) != 2:
            raise _err(422, f"range {c!r} must be [start_s, end_s]")
        for x in c:
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
                raise _err(422, f"range {c!r} must hold two finite numbers")
        out.append([float(c[0]), float(c[1])])
    return out


@router.post("/{dataset_id}/videos/{video_id}/start")
def start_video(dataset_id: str, video_id: str, body: StartBody, request: Request) -> dict:
    v = _video_or_404(request, dataset_id, video_id)
    if v["status"] != "awaiting_cut":
        raise _err(409, f"video is {v['status']}, not awaiting_cut")
    cuts = _check_cuts(body.cuts)
    if cuts:
        duration = v["duration_s"] or math.inf  # unknown duration: no clamp at the end
        try:
            cuts = [[s, e] for s, e in normalize_ranges(cuts, duration)]
        except BadCuts as e:
            raise _err(422, str(e)) from e
    store, settings = request.app.state.store, request.app.state.settings
    for slot in ("a", "b"):
        ok, where = models.available(models.current(store, slot), settings)
        if not ok:
            raise _err(422, f"model {slot.upper()} not found at {where}; fix it in Settings")
    request.app.state.runner.start_video(dataset_id, video_id, cuts)
    return {"ok": True, "cuts": cuts}


# --- clips
def _reason(c: dict) -> str | None:
    if c["status"] is None:  # between group and select: not decided yet
        return None
    return "disagree" if c.get("flag") == "disagree" else review_reason(c)


@router.get("/{dataset_id}/videos/{video_id}/clips")
def video_clips(dataset_id: str, video_id: str, request: Request) -> list[dict]:
    _video_or_404(request, dataset_id, video_id)
    store, data_dir = request.app.state.store, request.app.state.settings.data_dir
    map_file = cut_dir(data_dir, dataset_id, video_id) / "map.json"
    try:
        cmap = json.loads(map_file.read_text(encoding="utf-8")) if map_file.is_file() else None
    except ValueError:
        cmap = None
    out = []
    for c in store.clips(dataset_id, video_id=video_id):
        start, end = c["start_s"], c["end_s"]
        if cmap and start is not None and end is not None:
            o_start, o_end = to_orig(start, cmap), to_orig(end, cmap)
        else:
            o_start, o_end = start, end
        out.append({**c, "reason": _reason(c), "orig_start_s": o_start, "orig_end_s": o_end})
    return out
