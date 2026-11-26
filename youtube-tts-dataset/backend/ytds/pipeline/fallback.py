"""Fallback stage for clips where both STT models disagree: Whisper large-v3 translate -> LLM -> Bangla."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from ..llm import LlmUnavailable
from .audio16k import SR, load_16k
from .punct import translate_en_to_bn
from .transcribe import in_batches

TRANSLATE_MODEL = "large-v3"
COMPUTE = "int8_float16"


def load_whisper_translate() -> Callable[[list[np.ndarray]], list[str]]:
    """GPU loader registered as "whisper_translate" (Task 11)."""
    from faster_whisper import WhisperModel

    model = WhisperModel(TRANSLATE_MODEL, device="cuda", compute_type=COMPUTE)

    def translate(audios: list[np.ndarray]) -> list[str]:
        texts = []
        for a in audios:
            segments, _ = model.transcribe(a, language="bn", task="translate", beam_size=1)
            texts.append("".join(s.text for s in segments).strip())
        return texts

    return translate


def run_fallback(clips: list[dict], vocals_wav: Path, gpu, llm,
                 on_progress: Callable[[int, int], None]) -> None:
    """Process clips with status pending_llm. Whisper runs only for clips without text_en; the LLM then
    sets text_translated/text_final and status needs_review. If the LLM is down the clip stays pending_llm.
    Progress total is the number of pending clips."""
    pending = [c for c in clips if c.get("status") == "pending_llm"]
    if not pending:
        return
    total = len(pending)
    todo = [c for c in pending if c.get("text_en") is None]
    if todo:
        y = load_16k(vocals_wav)
        audios = [y[int(c["start_s"] * SR):int(c["end_s"] * SR)] for c in todo]
        for c, en in zip(todo, in_batches(gpu.use("whisper_translate"), audios)):
            c["text_en"] = en
    for i, c in enumerate(pending, 1):
        try:
            bn = translate_en_to_bn(c["text_en"], llm) if c["text_en"] else ""
        except LlmUnavailable:
            on_progress(i, total)
            continue
        c.update(text_translated=bn, text_final=bn, text_source="translated", status="needs_review")
        on_progress(i, total)
