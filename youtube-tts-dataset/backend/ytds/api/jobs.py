"""Dataset (job) and per-video routes."""
from __future__ import annotations

import re
import shutil
from pathlib import PureWindowsPath

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from ..pipeline import STAGES
from ..pipeline.download import expand_urls
from ..runner import video_dir

router = APIRouter(prefix="/api/datasets")
_BAD_NAME = re.compile(r'[<>:"|?*/\\\x00-\x1f]')


class CreateDataset(BaseModel):
    name: str
    speaker: str | None = None
    urls: str


class AddVideos(BaseModel):
    urls: str


class UpdateDataset(BaseModel):
    speaker: str


def valid_dataset_name(name: str) -> bool:
    """The name becomes the default export folder, so it must be one plain Windows file name."""
    return (bool(name) and name not in (".", "..") and not _BAD_NAME.search(name)
            and not PureWindowsPath(name).anchor and not name.endswith((".", " ")))


def dataset_or_404(request: Request, dataset_id: str) -> dict:
    ds = request.app.state.store.get_dataset(dataset_id)
    if ds is None:
        raise HTTPException(404, "dataset not found")
    return ds


def _not_exporting(request: Request, dataset_id: str) -> None:
    if dataset_id in request.app.state.exports:
        raise HTTPException(409, "dataset is exporting")


def _video_or_404(request: Request, dataset_id: str, video_id: str) -> dict:
    dataset_or_404(request, dataset_id)
    for v in request.app.state.store.list_videos(dataset_id):
        if v["id"] == video_id:
            return v
    raise HTTPException(404, "video not found")


def _expand(text: str) -> tuple[list[tuple[str, str]], list[dict]]:
    """Expand each line on its own so one bad URL does not fail the others."""
    pairs: list[tuple[str, str]] = []
    errors: list[dict] = []
    for line in text.splitlines():
        url = line.strip()
        if not url:
            continue
        try:
            pairs += expand_urls(url)
        except Exception as e:
            errors.append({"url": url, "error": str(e)})
    return pairs, errors


def _add(request: Request, dataset_id: str, pairs: list[tuple[str, str]], errors: list[dict]) -> dict:
    store = request.app.state.store
    added = sum(store.add_video(dataset_id, vid, url) for vid, url in pairs)
    if added:
        request.app.state.runner.enqueue(dataset_id)
    return {"dataset_id": dataset_id, "added": added, "duplicates": len(pairs) - added, "errors": errors}


@router.post("")
async def create_dataset(body: CreateDataset, request: Request) -> dict:
    store = request.app.state.store
    name = body.name.strip()
    if not valid_dataset_name(name):
        raise HTTPException(400, f"invalid dataset name {name!r}: use a plain folder name")
    if any(d["name"] == name for d in store.list_datasets()):
        raise HTTPException(409, f"dataset {name!r} already exists")
    pairs, errors = await run_in_threadpool(_expand, body.urls)
    if not pairs:
        raise HTTPException(400, {"error": "no usable URL", "errors": errors})
    dataset_id = store.create_dataset(name, (body.speaker or "").strip())
    return _add(request, dataset_id, pairs, errors)


@router.get("")
def list_datasets(request: Request) -> list[dict]:
    store = request.app.state.store
    return [{**d, "videos": len(store.list_videos(d["id"]))} for d in store.list_datasets()]


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str, request: Request) -> dict:
    store = request.app.state.store
    ds = dataset_or_404(request, dataset_id)
    clips = store.clips(dataset_id)
    videos = []
    for v in store.list_videos(dataset_id):
        mine = [c for c in clips if c["video_id"] == v["id"]]
        dropped = sum(1 for c in mine if c["status"] in ("dropped", "rejected"))
        videos.append({
            **v,
            "stages": {s: store.stage_done(dataset_id, v["id"], s) for s in STAGES},
            "kept": len(mine) - dropped,
            "dropped": dropped,
        })
    return {**ds, "videos": videos}


@router.put("/{dataset_id}")
def update_dataset(dataset_id: str, body: UpdateDataset, request: Request) -> dict:
    dataset_or_404(request, dataset_id)
    request.app.state.store.set_dataset_speaker(dataset_id, body.speaker.strip())
    return dataset_or_404(request, dataset_id)


@router.post("/{dataset_id}/videos")
async def add_videos(dataset_id: str, body: AddVideos, request: Request) -> dict:
    dataset_or_404(request, dataset_id)
    pairs, errors = await run_in_threadpool(_expand, body.urls)
    return _add(request, dataset_id, pairs, errors)


@router.post("/{dataset_id}/videos/{video_id}/retry")
def retry_video(dataset_id: str, video_id: str, request: Request) -> dict:
    _video_or_404(request, dataset_id, video_id)
    _not_exporting(request, dataset_id)
    request.app.state.runner.retry(dataset_id, video_id)
    return {"ok": True}


@router.post("/{dataset_id}/videos/{video_id}/skip")
def skip_video(dataset_id: str, video_id: str, request: Request) -> dict:
    _video_or_404(request, dataset_id, video_id)
    request.app.state.runner.skip(dataset_id, video_id)
    return {"ok": True}


@router.delete("/{dataset_id}/videos/{video_id}")
def delete_video(dataset_id: str, video_id: str, request: Request) -> dict:
    v = _video_or_404(request, dataset_id, video_id)
    _not_exporting(request, dataset_id)
    # Skip only flips the status; a stage already queued or running keeps going and still writes its outputs,
    # so the runner's in-flight set (not the status) decides whether the files and rows are free to delete.
    if v["status"] == "running" or request.app.state.runner.busy(dataset_id, video_id):
        raise HTTPException(409, "video is still processing; wait for it to finish or fail")
    request.app.state.store.delete_video(dataset_id, video_id)
    shutil.rmtree(video_dir(request.app.state.settings.data_dir, dataset_id, video_id), ignore_errors=True)
    return {"ok": True}
