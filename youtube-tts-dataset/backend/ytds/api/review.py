"""Review queue, clip edits and clip audio."""
from __future__ import annotations

import os
import threading
from typing import Literal

import soundfile as sf
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..runner import video_dir, vocals_path
from .jobs import dataset_or_404

router = APIRouter(prefix="/api/datasets")

QUEUE_STATUSES = ["flagged", "needs_review", "pending_llm"]
REVIEWED_STATUSES = ["approved", "rejected"]
Reason = Literal["cer", "translated", "pending_llm", "auto"]
_slice_lock = threading.Lock()  # a browser fires parallel Range requests for the same clip


class ClipPatch(BaseModel):
    text_final: str | None = None
    status: Literal["approved", "rejected"] | None = None
    text_pick: Literal["a", "b"] | None = None  # swap to this model's text (text_final follows unless given)


def review_reason(clip: dict) -> Reason:
    """Why a clip is (or was) in review: LLM down, translated text, flagged by the A/B check (only datasets
    transcribed before model B was removed have a CER), or accepted automatically."""
    if clip["status"] == "pending_llm":
        return "pending_llm"
    if clip["text_source"] == "translated":
        return "translated"
    if clip.get("cer") is not None:
        return "cer"
    return "auto"


def _clip_or_404(request: Request, dataset_id: str, clip_id: str) -> dict:
    dataset_or_404(request, dataset_id)
    clip = request.app.state.store.get_clip(dataset_id, clip_id)
    if clip is None:
        raise HTTPException(404, "clip not found")
    return clip


@router.get("/{dataset_id}/review")
def review_queue(dataset_id: str, request: Request, reason: Reason | None = None, video: str | None = None,
                 status: Literal["flagged", "needs_review", "pending_llm", "approved", "rejected", "auto"] | None = None,
                 ) -> dict:
    store = request.app.state.store
    dataset_or_404(request, dataset_id)
    clips = store.clips(dataset_id, video_id=video, statuses=[status] if status else QUEUE_STATUSES)
    out = [{**c, "reason": review_reason(c)} for c in clips]
    if reason:
        out = [c for c in out if c["reason"] == reason]
    return {
        "clips": out,
        "remaining": len(store.clips(dataset_id, statuses=QUEUE_STATUSES)),
        "reviewed": len(store.clips(dataset_id, statuses=REVIEWED_STATUSES)),
    }


@router.patch("/{dataset_id}/clips/{clip_id}")
def patch_clip(dataset_id: str, clip_id: str, body: ClipPatch, request: Request) -> dict:
    clip = _clip_or_404(request, dataset_id, clip_id)
    fields = body.model_dump(exclude_none=True)
    if "text_pick" in fields and "text_final" not in fields:
        fields["text_final"] = clip[f"text_{fields['text_pick']}"] or ""
    text = fields.get("text_final", clip["text_final"]) or ""
    if fields.get("status") == "approved" and not text.strip():
        raise HTTPException(422, "cannot approve a clip with empty text")
    request.app.state.store.update_clip(dataset_id, clip_id, **fields)
    clip = request.app.state.store.get_clip(dataset_id, clip_id)
    return {**clip, "reason": review_reason(clip)}


def _write_slice(vocals, out, start_s: float, end_s: float) -> None:
    with _slice_lock:
        info = sf.info(vocals)
        sr = info.samplerate
        start, stop = int(round(start_s * sr)), min(info.frames, int(round(end_s * sr)))
        if out.is_file() and sf.info(out).frames == stop - start:
            return  # cached slice still matches the clip span
        y, _ = sf.read(vocals, start=start, stop=stop, dtype="float32")
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".tmp.wav")
        sf.write(tmp, y, sr, subtype="PCM_16")
        os.replace(tmp, out)


@router.get("/{dataset_id}/clips/{clip_id}/audio")
def clip_audio(dataset_id: str, clip_id: str, request: Request) -> FileResponse:
    clip = _clip_or_404(request, dataset_id, clip_id)
    data_dir = request.app.state.settings.data_dir
    vocals = vocals_path(data_dir, dataset_id, clip["video_id"])
    if not vocals.is_file():
        raise HTTPException(404, "vocals stem not available yet")
    out = video_dir(data_dir, dataset_id, clip["video_id"]) / "clips" / f"{clip_id}.wav"
    _write_slice(vocals, out, clip["start_s"], clip["end_s"])
    return FileResponse(out, media_type="audio/wav")
