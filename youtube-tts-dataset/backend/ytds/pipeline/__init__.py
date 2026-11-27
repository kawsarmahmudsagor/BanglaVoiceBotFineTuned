"""Pipeline stages, in execution order (spec section 4.1)."""
from __future__ import annotations

from ytds.store import Stage

STAGES: tuple[Stage, ...] = (
    "download",
    "cut",
    "separate",
    "pieces",
    "transcribe_a",
    "transcribe_b",
    "punctuate_pieces",
    "group",
    "quality",
    "select",
    "punctuate",
)
