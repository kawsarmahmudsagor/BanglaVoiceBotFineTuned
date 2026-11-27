"""Transcription model registry: the model A / model B options of the Settings page (spec section 2.2).

An option id resolves to a local folder or a Hugging Face repo id. `available` and `source` never download or
convert anything; `resolve` (called only by the GPU loaders) converts the mozilla-ai checkpoint to CT2 once.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from .config import Settings

Slot = Literal["a", "b"]

SAYEDSHAUN_REPO = "SayedShaun/Qwen3-ASR-0.6B-Bengali"
MOZILLA_REPO = "mozilla-ai/whisper-large-v3-turbo-bn"
MOZILLA_COMPUTE = "float16"


@dataclass(frozen=True)
class ModelOption:
    id: str
    slot: Slot
    label: str
    setting: str | None = None  # Settings field holding a local folder or repo id
    repo: str | None = None     # fixed HF repo id


OPTIONS: list[ModelOption] = [
    ModelOption("qwen3-asr-bn-ft1", "a", "Qwen3-ASR 0.6B — Bengali.AI 200h fine-tune", setting="model_a_path"),
    ModelOption("qwen3-asr-bn-sayedshaun", "a", "Qwen3-ASR 0.6B Bengali (SayedShaun, current model)",
                repo=SAYEDSHAUN_REPO),
    ModelOption("whisper-bn-ft1", "b", "Whisper large-v3-turbo-bn — Bengali.AI 200h fine-tune (r-854260e0)",
                setting="model_b_path"),
    ModelOption("whisper-bn-mozilla", "b", "Whisper large-v3-turbo-bn (mozilla-ai, the earlier model B)",
                repo=MOZILLA_REPO),
]
BY_ID = {o.id: o for o in OPTIONS}
DEFAULTS: dict[str, str] = {"a": "qwen3-asr-bn-ft1", "b": "whisper-bn-ft1"}


def options(slot: Slot) -> list[ModelOption]:
    return [o for o in OPTIONS if o.slot == slot]


def _is_repo_id(value: str) -> bool:
    return value.count("/") == 1 and ":" not in value and "\\" not in value and not value.startswith("/")


def source(option_id: str, settings: Settings) -> str:
    """The configured folder or repo id of an option, without converting anything (provenance, Settings page)."""
    o = BY_ID[option_id]
    return getattr(settings, o.setting) if o.setting else o.repo


def available(option_id: str, settings: Settings) -> tuple[bool, str]:
    """(ok, expected path or repo id): an existing folder or a repo-id-shaped value counts as available."""
    value = source(option_id, settings)
    return (Path(value).is_dir() or _is_repo_id(value)), value


def ct2_dir(ckpt: str, data_dir: Path, compute_type: str = MOZILLA_COMPUTE) -> Path:
    """CT2 conversion of the transformers Whisper checkpoint `ckpt`, made once and cached under
    <data_dir>/models/ (restored from before 5c05bfa; converted in-process, Windows has no converter script)."""
    out = Path(data_dir) / "models" / f"{ckpt.strip('/').replace('/', '--')}-{compute_type}"
    if (out / "model.bin").exists():
        return out
    from ctranslate2.converters import TransformersConverter

    tmp = out.with_name(out.name + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    TransformersConverter(ckpt, copy_files=["preprocessor_config.json"]).convert(
        str(tmp), quantization=compute_type, force=True
    )
    if not (tmp / "tokenizer.json").exists():  # some repos ship only vocab.json/merges.txt
        from transformers import WhisperTokenizerFast

        WhisperTokenizerFast.from_pretrained(ckpt).backend_tokenizer.save(str(tmp / "tokenizer.json"))
    tmp.rename(out)
    return out


def resolve(option_id: str, settings: Settings) -> str:
    """What the loader opens: a folder or repo id. The mozilla option is converted to CT2 float16 on first use."""
    if option_id == "whisper-bn-mozilla":
        return str(ct2_dir(MOZILLA_REPO, settings.data_dir))
    return source(option_id, settings)


def current(store, slot: Slot) -> str:
    """The chosen option of a slot (setting model_a / model_b); unknown or wrong-slot values give the default."""
    value = store.get_setting(f"model_{slot}")
    o = BY_ID.get(value or "")
    return o.id if o and o.slot == slot else DEFAULTS[slot]


class WhisperB:
    """The cached model B object: `__call__(audio16k, pieces) -> [{text, tokens, compression, retry_failed}]`."""

    def __init__(self, decode: Callable, option_id: str):
        self._decode, self.option_id = decode, option_id

    def __call__(self, audio16k, pieces: list[dict]) -> list[dict]:
        return self._decode(audio16k, pieces)


def loader(slot: Slot, store, settings: Settings) -> Callable[[], Any]:
    """GPU worker loader for "model_a" / "model_b". The option is read when the model loads, so a Settings change
    does not affect an already loaded model; the loaded object carries `.option_id`."""

    def load() -> Any:
        option_id = current(store, slot)
        path = resolve(option_id, settings)
        if slot == "a":
            from .pipeline import transcribe

            model = transcribe.load_model_a(path)
            model.option_id = option_id
            return model
        from .pipeline import whisper_b

        return WhisperB(whisper_b.load_whisper(path), option_id)

    return load
