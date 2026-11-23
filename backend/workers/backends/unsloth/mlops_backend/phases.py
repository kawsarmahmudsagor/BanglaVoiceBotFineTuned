"""Phase handlers: dispatch on the base model's family (chat -> llm, whisper -> asr)."""
from __future__ import annotations

import importlib
import sys

from app.core.backend_kit import Handler
from app.core.contract import Phase, RunConfig, RunDir

from .models import MODELS

BACKEND = "unsloth"
_FAMILY_MODULE = {"chat": "llm", "whisper": "asr"}


def _dispatch(phase: Phase) -> Handler:
    def handler(rd: RunDir, cfg: RunConfig, rep) -> int:
        info = MODELS.get(cfg.base_model)
        if info is None or info.family not in _FAMILY_MODULE:
            print(f"unsupported base_model '{cfg.base_model}' for backend {BACKEND}", file=sys.stderr)
            return 1
        # Family modules import torch/unsloth at call time only.
        try:
            mod = importlib.import_module(f".{_FAMILY_MODULE[info.family]}", __package__)
        except ImportError as e:
            print(f"backend {BACKEND}: {info.family} support not installed ({e})", file=sys.stderr)
            return 1
        fn = getattr(mod, phase.value, None)
        if fn is None:
            print(f"{phase.value} is not implemented for family '{info.family}'", file=sys.stderr)
            return 1
        return fn(rd, cfg, rep)

    return handler


HANDLERS: dict[Phase, Handler] = {p: _dispatch(p) for p in Phase}
