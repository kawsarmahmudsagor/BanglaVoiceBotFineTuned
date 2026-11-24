"""Supported base models. Keys are the `base_model` enum in capabilities.json."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    task: str
    family: str
    gated: bool


MODELS: dict[str, ModelInfo] = {
    "FunAudioLLM/Fun-CosyVoice3-0.5B-2512": ModelInfo("tts", "cosyvoice", False),
}
