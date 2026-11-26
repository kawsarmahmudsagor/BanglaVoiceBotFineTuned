"""CER between the two STT transcripts and the routing decision it drives."""
from __future__ import annotations

from typing import Literal

from app.core.text import bn_normalize

from ..config import Thresholds

Route = Literal["auto", "flagged", "fallback"]


def _chars(text: str) -> str:
    return bn_normalize(text, mode="asr").replace(" ", "")


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    r, h = _chars(ref), _chars(hyp)
    if not r and not h:
        return 0.0
    return _edit_distance(r, h) / max(len(r), 1)


def route(text_a: str, text_b: str, th: Thresholds) -> tuple[Route, float]:
    """Model A is the reference. "fallback" is stored as status pending_llm until the fallback stage runs."""
    if not _chars(text_a):
        return "fallback", 1.0
    c = cer(text_a, text_b)
    if c <= th.cer_auto:
        return "auto", c
    if c <= th.cer_fallback:
        return "flagged", c
    return "fallback", c
