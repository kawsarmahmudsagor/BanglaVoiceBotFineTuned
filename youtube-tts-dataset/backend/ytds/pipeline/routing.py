"""Routing of a model A transcript: kept as is ("auto") or sent to the English + LLM fallback.

There is one STT model, so the checks look at the transcript alone: empty output, too little Bangla script,
a speaking rate no person has, or a word repeated in a loop (a typical ASR hallucination).
"""
from __future__ import annotations

from typing import Literal

from app.core.text import bn_normalize

from ..config import Thresholds

Route = Literal["auto", "flagged", "fallback"]


def _chars(text: str) -> str:
    return bn_normalize(text, mode="asr").replace(" ", "")


def bn_ratio(text: str) -> float:
    """Share of Bangla-block characters among the letters and marks in `text` (digits and punctuation ignored)."""
    letters = [ch for ch in text if ch.isalpha() or "ঀ" <= ch <= "৿"]
    if not letters:
        return 0.0
    return sum("ঀ" <= ch <= "৿" for ch in letters) / len(letters)


def longest_repeat(text: str) -> int:
    """Length of the longest run of one word repeated back to back."""
    best = run = 0
    prev = None
    for w in text.split():
        run = run + 1 if w == prev else 1
        prev = w
        best = max(best, run)
    return best


def route(text_a: str, duration_s: float, th: Thresholds) -> Route:
    """"fallback" is stored as status pending_llm until the fallback stage runs."""
    chars = _chars(text_a)
    if not chars:
        return "fallback"
    if bn_ratio(text_a) < th.min_bn_ratio:
        return "fallback"
    cps = len(chars) / max(duration_s, 0.1)
    if cps < th.min_chars_per_s or cps > th.max_chars_per_s:
        return "fallback"
    if longest_repeat(text_a) >= th.max_word_repeat:
        return "fallback"
    return "auto"
