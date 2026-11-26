"""Settings lookup: process env > youtube-tts-dataset/.env > repo-root .env."""
from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel

SUB_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SUB_ROOT.parent


class Thresholds(BaseModel):
    min_bn_ratio: float = 0.5        # below this share of Bangla script, the transcript goes to the fallback
    min_chars_per_s: float = 3.0     # plausible speaking rate, in Bangla characters per second of audio
    max_chars_per_s: float = 30.0
    max_word_repeat: int = 6         # one word this many times in a row is a hallucination loop
    dnsmos_min: float = 3.0
    speaker_sim_min: float = 0.60
    clip_min_s: float = 2.0
    clip_max_s: float = 15.0
    merge_gap_s: float = 0.30
    demucs_segment_s: float = 7.0


class Settings(BaseModel):
    ollama_host: str = "http://localhost:11434"
    ollama_api_key: str = ""
    llm_model: str = "gemma4:31b-cloud"
    data_dir: Path = SUB_ROOT / "var"
    export_dir: Path = REPO_ROOT / "var" / "import"
    port: int = 8765
    hf_token: str = ""


def _file_values(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    return {k: v for k, v in dotenv_values(path).items() if v is not None}


def _resolve_dir(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


def load_settings(
    env: Mapping[str, str] | None = None,
    sub_env: Path | None = None,
    root_env: Path | None = None,
) -> Settings:
    if env is None:
        env = os.environ
    if sub_env is None:
        sub_env = SUB_ROOT / ".env"
    if root_env is None:
        root_env = REPO_ROOT / ".env"
    merged: dict[str, str] = {**_file_values(root_env), **_file_values(sub_env), **env}

    kw: dict = {}
    for key, field in [
        ("OLLAMA_HOST", "ollama_host"),
        ("OLLAMA_API_KEY", "ollama_api_key"),
        ("LLM_MODEL", "llm_model"),
        ("YTDS_PORT", "port"),
        ("HF_TOKEN", "hf_token"),
    ]:
        if merged.get(key):
            kw[field] = merged[key]
    if merged.get("YTDS_DATA_DIR"):
        kw["data_dir"] = _resolve_dir(merged["YTDS_DATA_DIR"])
    if merged.get("YTDS_EXPORT_DIR"):
        kw["export_dir"] = _resolve_dir(merged["YTDS_EXPORT_DIR"])
    return Settings(**kw)


def export_hf_token(settings: Settings, environ: MutableMapping[str, str] | None = None) -> None:
    """huggingface_hub reads HF_TOKEN from the process environment only, so a token that came from a .env file
    is copied there. A token already set in the environment wins."""
    if environ is None:
        environ = os.environ
    if settings.hf_token:
        environ.setdefault("HF_TOKEN", settings.hf_token)
