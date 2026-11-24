"""Supported base models. Keys are the `base_model` enum in capabilities.json."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    task: str
    family: str
    gated: bool


MODELS: dict[str, ModelInfo] = {
    "openai/whisper-large-v3-turbo": ModelInfo("asr", "whisper", False),
    "mozilla-ai/whisper-large-v3-turbo-bn": ModelInfo("asr", "whisper", False),
    "facebook/wav2vec2-xls-r-300m": ModelInfo("asr", "ctc", False),
    "facebook/mms-1b-all": ModelInfo("asr", "ctc-mms", False),
    "ai4bharat/indic-parler-tts": ModelInfo("tts", "parler", True),
}
