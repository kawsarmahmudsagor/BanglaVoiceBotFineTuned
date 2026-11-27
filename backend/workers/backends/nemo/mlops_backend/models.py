"""Supported base models. Keys are the `base_model` enum in capabilities.json."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    task: str
    family: str
    gated: bool


MODELS: dict[str, ModelInfo] = {
    "ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large": ModelInfo("asr", "nemo-manifest", True),
    "bengaliAI/BanglaConformer": ModelInfo("asr", "nemo-manifest", False),
    "SayedShaun/stt_bn_fastconformer_hybrid_large_v2": ModelInfo("asr", "nemo-manifest", False),
}
