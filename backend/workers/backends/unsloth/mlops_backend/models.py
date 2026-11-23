"""Supported base models. Keys are the `base_model` enum in capabilities.json."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    task: str
    family: str
    gated: bool
    chat_template: str | None = None


MODELS: dict[str, ModelInfo] = {
    "Qwen/Qwen3-4B-Instruct-2507": ModelInfo("llm-sft", "chat", False, "qwen3"),
    "google/gemma-3-4b-it": ModelInfo("llm-sft", "chat", True, "gemma3"),
    "hishab/titulm-llama-3.2-3b-v2.0": ModelInfo("llm-sft", "chat", False, "llama-3.2"),
    "openai/whisper-large-v3-turbo": ModelInfo("asr", "whisper", False),
    "mozilla-ai/whisper-large-v3-turbo-bn": ModelInfo("asr", "whisper", False),
}
