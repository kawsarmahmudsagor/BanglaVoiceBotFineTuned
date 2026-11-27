"""Transcribe stage: model A (Qwen3-ASR Bengali), batched on the GPU.

Model loaders are plain functions; the GPU worker's loaders are registered by the app (Task 11, ruling R2).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import Thresholds
from .audio16k import SR, load_16k
from .routing import route

Transcriber = Callable[[list[np.ndarray]], list[str]]

MODEL_A = "SayedShaun/Qwen3-ASR-0.6B-Bengali"
BATCH = 8

# --- Ported from backend/workers/backends/qwen-asr/mlops_backend/prompt.py and prepare/qwen_asr.py
# (that venv is Linux-only, so the helpers are copied rather than imported).
LANGUAGE = "Bengali"
MARKER = "<asr_text>"
PREFIX = f"language {LANGUAGE}{MARKER}"
TOKENS_PER_SECOND = 30  # model card: about 30 tokens per second of audio
EXTRA_NEW_TOKENS = 16
MAX_SECONDS = 30


def _conversation(audio_ref: str = "audio.wav") -> list[dict]:
    return [
        {"role": "user", "content": [{"type": "audio", "path": audio_ref}]},
        {"role": "assistant", "content": [{"type": "text", "text": PREFIX}]},
    ]


def prompt_text(processor) -> str:
    """The rendered prompt up to and including PREFIX, with the audio placeholder unexpanded."""
    s = processor.apply_chat_template(_conversation(), tokenize=False, continue_final_message=True)
    if isinstance(s, list):
        s = s[0]
    if not s.endswith(PREFIX):
        raise ValueError(f"chat template prompt does not end with the prefix {PREFIX!r}: {s[-80:]!r}")
    return s


def generation_inputs(processor, prompt: str, audios: list, device=None, dtype=None):
    """Inference inputs for a batch of 16 kHz clips: the prompt per clip, left padded for generation."""
    b = processor(text=[prompt] * len(audios), audio=audios, sampling_rate=SR, padding=True,
                  padding_side="left", return_tensors="pt")
    b.pop("num_audio_tokens", None)
    return b.to(device, dtype) if device is not None else b


def max_new_tokens(max_audio_s: float) -> int:
    return int(TOKENS_PER_SECOND * max_audio_s) + EXTRA_NEW_TOKENS


def split_transcript(decoded: str) -> str:
    """Generated text -> transcript: the part after the marker (the whole text when the marker is absent)."""
    return strip_glosses(decoded.split(MARKER, 1)[-1])
# --- end of port

_GLOSS = re.compile(r"\s*\{[^{}]*\}")


def strip_glosses(text: str) -> str:
    """Drop the English glosses model A sometimes adds after loanwords ("স্কিল {Skill}" -> "স্কিল")."""
    return re.sub(r"\s+", " ", _GLOSS.sub("", text)).strip()


def in_batches(model: Transcriber, audios: list[np.ndarray],
               on_batch: Callable[[int], None] | None = None) -> list[str]:
    """Run `model` over `audios` BATCH clips at a time; `on_batch(done)` after each batch."""
    out: list[str] = []
    for lo in range(0, len(audios), BATCH):
        out.extend(model(audios[lo:lo + BATCH]))
        if on_batch is not None:
            on_batch(len(out))
    return out


def load_model_a() -> Transcriber:
    import torch
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_A)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        MODEL_A, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda").eval()
    prompt = prompt_text(processor)

    def batch(audios: list[np.ndarray]) -> list[str]:
        inputs = generation_inputs(processor, prompt, audios, model.device, model.dtype)
        longest = min(max(len(a) for a in audios) / SR, MAX_SECONDS)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens(longest), do_sample=False)
        texts = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return [split_transcript(t) for t in texts]

    return lambda audios: in_batches(batch, audios)


def transcribe_clips(clips: list[dict], vocals_wav: Path, gpu, on_progress: Callable[[int, int], None],
                     th: Thresholds | None = None) -> None:
    """All live clips through model A; sets text_a, status, text_source (text_b and cer stay empty).
    A "broken" route is stored as status pending_llm. Progress total is the number of live clips."""
    th = th or Thresholds()
    live = [c for c in clips if c.get("status") != "dropped"]
    if not live:
        return
    y = load_16k(vocals_wav)
    audios = [y[int(c["start_s"] * SR):int(c["end_s"] * SR)] for c in live]
    n = len(live)
    texts_a = in_batches(gpu.use("model_a"), audios, lambda d: on_progress(d, n))
    for c, a in zip(live, texts_a):
        status = route(a, c["end_s"] - c["start_s"], th)
        c.update(text_a=a, text_b=None, cer=None, text_source="stt",
                 status="pending_llm" if status == "broken" else status)
