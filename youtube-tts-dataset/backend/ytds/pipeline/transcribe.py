"""Model A (Qwen3-ASR Bengali, transformers): loader and piece transcription, batched on the GPU.

The GPU worker's "model_a" loader is built by ytds.models.loader, which resolves the chosen option to a path.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np

from .audio16k import SR, load_16k

Transcriber = Callable[[list[np.ndarray]], list[str]]

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


def load_model_a(path: str) -> Transcriber:
    """A Qwen3-ASR checkpoint (local safetensors folder or HF repo id), bf16 on CUDA."""
    import torch
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    processor = AutoProcessor.from_pretrained(path)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="sdpa"
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


def transcribe_pieces_a(pieces: list[dict], vocals_wav: Path, model: Transcriber,
                        on_progress: Callable[[int, int], None]) -> list[str]:
    """Model A text per piece (start_s/end_s on the vocals), in piece order. Progress total is len(pieces)."""
    if not pieces:
        return []
    y = load_16k(vocals_wav)
    audios = [y[int(p["start_s"] * SR):int(p["end_s"] * SR)] for p in pieces]
    n = len(pieces)
    return in_batches(model, audios, lambda d: on_progress(d, n))
