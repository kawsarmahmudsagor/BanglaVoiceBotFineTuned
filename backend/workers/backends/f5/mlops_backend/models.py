"""Supported base models. Keys are the `base_model` enum in capabilities.json."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    task: str
    family: str
    gated: bool


MODELS: dict[str, ModelInfo] = {
    "ai4bharat/IndicF5": ModelInfo("tts", "f5", True),
}
