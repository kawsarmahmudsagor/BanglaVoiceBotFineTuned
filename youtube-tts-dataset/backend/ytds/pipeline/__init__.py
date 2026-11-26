"""Pipeline stages, in execution order."""
from __future__ import annotations

from ytds.store import Stage

STAGES: tuple[Stage, ...] = (
    "download",
    "separate",
    "segment",
    "quality",
    "transcribe",
    "fallback",
    "punctuate",
)
