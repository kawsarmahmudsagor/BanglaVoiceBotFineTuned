"""Supported base models. Keys are the `base_model` enum in capabilities.json."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    task: str
    family: str
    gated: bool


MODELS: dict[str, ModelInfo] = {
    "SayedShaun/Qwen3-ASR-0.6B-Bengali": ModelInfo("asr", "qwen-asr", False),
    # Comparison only (Bengali is not in its language list). This is the transformers-format repo: the original
    # Qwen/Qwen3-ASR-0.6B ships a chat template that drops a prefilled assistant turn, so continue_final_message fails.
    "Qwen/Qwen3-ASR-0.6B-hf": ModelInfo("asr", "qwen-asr", False),
}
