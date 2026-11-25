"""Prompt format, label masking and the training collator (spec 17.4). Pure Python: no torch/transformers import.

Training and inference use the model card's chat format: a user turn holding the audio, then an assistant turn
that starts with PREFIX. The rendered prompt comes from the processor's own chat template
(`apply_chat_template(..., continue_final_message=True)`), with one audio placeholder that the processor expands
to the clip's audio token count. Training appends the transcript and EOS to that prompt; labels are -100 up to and
including the `<asr_text>` marker (the last token of PREFIX) and on padding, so loss covers transcript + EOS only.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

LANGUAGE = "Bengali"
MARKER = "<asr_text>"
PREFIX = f"language {LANGUAGE}{MARKER}"
IGNORE = -100
SAMPLE_RATE = 16000
TOKENS_PER_SECOND = 30  # model card: about 30 tokens per second of audio
EXTRA_NEW_TOKENS = 16


def conversation(audio_ref: str = "audio.wav") -> list[dict]:
    """One request in the model card's chat format, the assistant turn prefilled with PREFIX."""
    return [
        {"role": "user", "content": [{"type": "audio", "path": audio_ref}]},
        {"role": "assistant", "content": [{"type": "text", "text": PREFIX}]},
    ]


def prompt_text(processor) -> str:
    """The rendered prompt up to and including PREFIX, with the audio placeholder unexpanded.
    The template is static apart from the audio placeholder, so one string serves every clip."""
    s = processor.apply_chat_template(conversation(), tokenize=False, continue_final_message=True)
    if isinstance(s, list):
        s = s[0]
    if not s.endswith(PREFIX):
        raise ValueError(f"chat template prompt does not end with the prefix {PREFIX!r}: {s[-80:]!r}")
    return s


def training_text(prompt: str, transcript: str, eos: str) -> str:
    return prompt + transcript + eos


def mask_labels(input_ids: Sequence[Sequence[int]], attention_mask: Sequence[Sequence[int]],
                marker_id: int) -> list[list[int]]:
    """Labels for a batch: the token ids, with -100 on every position up to and including the first marker
    token (system, user/audio turn, assistant header and PREFIX) and on padding (attention_mask 0)."""
    out: list[list[int]] = []
    for row, mask in zip(input_ids, attention_mask):
        row = [int(t) for t in row]
        try:
            cut = row.index(marker_id)
        except ValueError:
            raise ValueError(f"marker token {marker_id} ({MARKER}) not found in a training row") from None
        labels = [IGNORE if (i <= cut or not int(m)) else t for i, (t, m) in enumerate(zip(row, mask))]
        out.append(labels)
    return out


def split_transcript(decoded: str) -> str:
    """Generated text -> transcript: the part after the marker (the whole text when the marker is absent)."""
    return decoded.split(MARKER, 1)[-1].strip()


def max_new_tokens(max_audio_s: float) -> int:
    return int(TOKENS_PER_SECOND * max_audio_s) + EXTRA_NEW_TOKENS


def _default_loader(path):
    from app.core.backend_kit.audio import load_audio

    return load_audio(path, SAMPLE_RATE)


def _like(ref, rows: list[list[int]]):
    """`rows` as the same kind of container as `ref` (a torch tensor when the processor returned tensors)."""
    return ref.new_tensor(rows) if hasattr(ref, "new_tensor") else rows


class QwenAsrCollator:
    """Batches prepared rows ({audio_path, text}): loads audio, builds the chat inputs with the processor
    (right padding) and masks labels to transcript + EOS."""

    def __init__(self, processor, loader: Callable = _default_loader):
        self.processor = processor
        self.loader = loader
        self.prompt = prompt_text(processor)
        tok = processor.tokenizer
        self.eos = tok.eos_token
        self.marker_id = int(tok.convert_tokens_to_ids(MARKER))

    def __call__(self, features: list[dict]) -> dict:
        audios = [self.loader(str(f["audio_path"])) for f in features]
        texts = [training_text(self.prompt, f["text"], self.eos) for f in features]
        b = self.processor(text=texts, audio=audios, sampling_rate=SAMPLE_RATE, padding=True,
                           padding_side="right", return_tensors="pt")
        batch = {k: v for k, v in dict(b).items() if k != "num_audio_tokens"}
        ids, mask = batch["input_ids"], batch["attention_mask"]
        rows = ids.tolist() if hasattr(ids, "tolist") else ids
        masks = mask.tolist() if hasattr(mask, "tolist") else mask
        batch["labels"] = _like(ids, mask_labels(rows, masks, self.marker_id))
        return batch


def generation_inputs(processor, prompt: str, audios: list, device=None, dtype=None):
    """Inference inputs for a batch of 16 kHz clips: the prompt per clip, left padded for generation."""
    b = processor(text=[prompt] * len(audios), audio=audios, sampling_rate=SAMPLE_RATE, padding=True,
                  padding_side="left", return_tensors="pt")
    b.pop("num_audio_tokens", None)
    return b.to(device, dtype) if device is not None else b
