"""LLM punctuation with a guard: the model may only change punctuation, never the words."""
from __future__ import annotations

import re

from app.core.text import bn_normalize

from ..llm import PUNCT_SYSTEM, LlmUnavailable

_PUNCT = re.compile(r"[।?!,;:\"'“”‘’()\-—.]")


def _words(text: str) -> list[str]:
    return bn_normalize(_PUNCT.sub("", text), mode="asr").split()


def words_unchanged(original: str, punctuated: str) -> bool:
    return _words(original) == _words(punctuated)


def punctuate(text: str, llm) -> tuple[str, bool]:
    """Returns (text, no_punct). no_punct is True when the original is kept because the LLM was down or its
    output was rejected; empty text has nothing to punctuate and is not a failure."""
    if not text.strip():
        return text, False
    try:
        out = llm.chat(PUNCT_SYSTEM, text)
    except LlmUnavailable:
        return text, True
    if not out or not words_unchanged(text, out):
        return text, True
    return out, False

