"""Qwen3-ASR for the playground: one model wrapper for the file tab and a RealtimeSTT engine for the live tab.

Inference is the same as the qwen-asr backend's eval (backend/workers/backends/qwen-asr/mlops_backend/prompt.py,
loaded from that file so the prompt cannot drift): the model card chat prompt with the assistant turn prefilled
with `language Bengali<asr_text>`, greedy decoding, bf16.

Live tab: RealtimeSTT's engine registry is a fixed dict and its built-in `qwen3_asr` engine needs the `qwen-asr`
pip package and another prompt, so `register()` adds ENGINE to the registry. The recorder builds the engine in a
spawned child process, which re-imports the main module (stt_ui.py) and so calls `register()` again there.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ENGINE = "playground_qwen3_asr"
SAMPLE_RATE = 16000
_PROMPT_PY = Path(__file__).resolve().parent.parent / "backend/workers/backends/qwen-asr/mlops_backend/prompt.py"


def _prompt():
    """The backend's prompt module (pure Python), imported by file path."""
    name = "_qwen_asr_backend_prompt"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _PROMPT_PY)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def is_qwen(ckpt: str) -> bool:
    """Whether `ckpt` (HF repo id or dir) is a Qwen3-ASR model. A CT2 dir (model.bin) is Whisper."""
    p = Path(ckpt)
    if p.is_dir() and (p / "model.bin").exists():
        return False
    try:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(ckpt).model_type.startswith("qwen3_asr")
    except Exception:  # noqa: BLE001 - unknown/offline: fall back to the name
        return "qwen3-asr" in ckpt.lower()


class QwenAsr:
    """Qwen3ASRForConditionalGeneration + its processor, transcribing one 16 kHz float32 clip at a time."""

    def __init__(self, ckpt: str, device: str = "cuda"):
        import torch
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(ckpt)
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.model = Qwen3ASRForConditionalGeneration.from_pretrained(
            ckpt, dtype=dtype, attn_implementation="sdpa").to(device).eval()
        self.prompt = _prompt().prompt_text(self.processor)

    def transcribe(self, wav: np.ndarray) -> str:
        import torch

        p = _prompt()
        wav = np.asarray(wav, dtype=np.float32)
        inputs = p.generation_inputs(self.processor, self.prompt, [wav], self.model.device, self.model.dtype)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=p.max_new_tokens(len(wav) / SAMPLE_RATE),
                                      do_sample=False)
        text = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return p.split_transcript(text)


def register() -> None:
    """Add ENGINE to RealtimeSTT's engine registry (in whichever process imports this)."""
    from RealtimeSTT.transcription_engines import factory

    factory.ENGINE_CLASS_PATHS.setdefault(ENGINE, ("qwen_engine", "PlaygroundQwenEngine"))


def _engine_base():
    from RealtimeSTT.transcription_engines.base import BaseTranscriptionEngine

    return BaseTranscriptionEngine


class PlaygroundQwenEngine(_engine_base()):
    """RealtimeSTT engine over QwenAsr. The language is fixed to Bengali by the prompt; `language` is ignored."""

    engine_name = ENGINE

    def __init__(self, config):
        super().__init__(config)
        self.asr = QwenAsr(config.model, config.device)

    def warmup(self, audio):
        self.asr.transcribe(audio)

    def transcribe(self, audio, language=None, use_prompt=True, **kwargs):
        from RealtimeSTT.transcription_engines.base import TranscriptionInfo, TranscriptionResult

        text = self.asr.transcribe(self._normalize_audio(audio))
        return TranscriptionResult(text=text, info=TranscriptionInfo(language="bn", language_probability=1.0))
