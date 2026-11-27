"""Dataset stats and export."""
from __future__ import annotations

import threading
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..exporter import export_dataset, exportable
from ..splits import assign_splits
from .jobs import dataset_or_404
from .review import QUEUE_STATUSES

router = APIRouter(prefix="/api/datasets")
_exports_lock = threading.Lock()


class ExportRequest(BaseModel):
    out_dir: str | None = None
    skip_unreviewed_translated: bool = False  # export without unreviewed translated clips instead of 409


def _hours(clips: list[dict]) -> float:
    return sum(c["end_s"] - c["start_s"] for c in clips) / 3600


@router.get("/{dataset_id}/stats")
def stats(dataset_id: str, request: Request) -> dict:
    store = request.app.state.store
    dataset_or_404(request, dataset_id)
    kept, _ = exportable(store, dataset_id)
    dropped = Counter(c["drop_reason"] or "other" for c in store.clips(dataset_id, statuses=["dropped"]))
    dropped["rejected"] = len(store.clips(dataset_id, statuses=["rejected"]))
    dropped["unreviewed"] = len(store.clips(dataset_id, statuses=QUEUE_STATUSES))
    videos = []
    for v in store.list_videos(dataset_id):
        mine = [c for c in kept if c["video_id"] == v["id"]]
        lost = Counter(c["drop_reason"] or c["status"]
                       for c in store.clips(dataset_id, video_id=v["id"], statuses=["dropped", "rejected"]))
        videos.append({"video_id": v["id"], "title": v["title"], "status": v["status"],
                       "clips_kept": len(mine), "hours_kept": _hours(mine), "dropped": dict(lost)})
    split_of = assign_splits(kept)
    splits = {}
    for name in ("train", "val", "test"):
        part = [c for c in kept if split_of[c["id"]] == name]
        splits[name] = {"clips": len(part), "hours": _hours(part)}
    return {"clips_kept": len(kept), "hours_kept": _hours(kept), "dropped": dict(dropped),
            "videos": videos, "splits": splits}


def _run_export(app, ds: dict, out_dir: Path, skip: bool) -> None:
    bus, dataset_id = app.state.bus, ds["id"]

    def on_progress(done: int, total: int) -> None:
        bus.publish({"type": "progress", "dataset_id": dataset_id, "stage": "export", "done": done, "total": total})

    try:
        summary = export_dataset(app.state.store, dataset_id, out_dir, on_progress,
                                 data_dir=app.state.settings.data_dir, skip_unreviewed_translated=skip,
                                 settings=app.state.settings)
    except Exception as e:
        bus.publish({"type": "stage_failed", "dataset_id": dataset_id, "stage": "export", "error": str(e)})
    else:
        bus.publish({"type": "stage_done", "dataset_id": dataset_id, "stage": "export", "summary": summary})
    finally:
        app.state.exports.discard(dataset_id)


@router.post("/{dataset_id}/export", status_code=202)
def export(dataset_id: str, body: ExportRequest, request: Request):
    app = request.app
    ds = dataset_or_404(request, dataset_id)
    kept, unreviewed = exportable(app.state.store, dataset_id)
    if unreviewed and not body.skip_unreviewed_translated:
        return JSONResponse({"unreviewed": unreviewed}, status_code=409)
    if not kept:
        return JSONResponse({"error": "nothing to export"}, status_code=409)
    out_dir = Path(body.out_dir) if body.out_dir else Path(app.state.settings.export_dir) / ds["name"]
    if out_dir.exists() and not any((out_dir / f).is_file() for f in ("metadata.csv", "curation.json")):
        # export_dataset replaces out_dir wholesale: only ever replace a previous export
        return JSONResponse({"error": "out_dir exists and is not a previous export"}, status_code=409)
    with _exports_lock:
        if dataset_id in app.state.exports:
            return JSONResponse({"error": "export already running"}, status_code=409)
        app.state.exports.add(dataset_id)
    threading.Thread(target=_run_export, args=(app, ds, out_dir, body.skip_unreviewed_translated),
                     name=f"ytds-export-{dataset_id}", daemon=True).start()
    return {"out_dir": str(out_dir)}


@router.post("/{dataset_id}/retry-llm", status_code=202)
def retry_llm(dataset_id: str, request: Request) -> dict:
    """Rerun punctuation and translation for clips that waited on the LLM (spec section 8)."""
    dataset_or_404(request, dataset_id)
    request.app.state.runner.retry_llm(dataset_id)
    return {"ok": True}
