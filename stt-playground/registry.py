"""Model registry for the playground dropdown: the base models plus every checkpoint the training platform wrote.

Fine-tuned models are discovered from the runs root (default /srv/mlops/runs, override with STT_RUNS_ROOT) on
each call, so a new run shows up after a refresh. For each ASR run it lists the export (`ct2` for Whisper,
`safetensors` for Qwen3-ASR, `nemo` for the NeMo models, with the run's eval WER), the model written after the run
finished. Intermediate `checkpoint-N` / `step-N` dirs are not listed; a run without an export falls back to its
`final` checkpoint.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

RUNS_ROOT = Path(os.environ.get("STT_RUNS_ROOT", "/srv/mlops/runs"))

BASE_MODELS = {
    "Base · Whisper large-v3-turbo (openai)": "openai/whisper-large-v3-turbo",
    "Base · Whisper large-v3-turbo Bengali (mozilla-ai, base of the Whisper fine-tunes)":
        "mozilla-ai/whisper-large-v3-turbo-bn",
    "Base · Qwen3-ASR 0.6B Bengali (SayedShaun, base of the Qwen fine-tunes)": "SayedShaun/Qwen3-ASR-0.6B-Bengali",
    "Base · Bengali FastConformer hybrid RNN-T/CTC (SayedShaun, base of the NeMo fine-tunes)":
        "SayedShaun/stt_bn_fastconformer_hybrid_large_v2",
}


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _family(base_model: str, backend: str) -> str:
    if backend == "nemo":
        return "NeMo"
    return "Qwen3-ASR" if backend == "qwen-asr" or "qwen3-asr" in base_model.lower() else "Whisper"


def _run_entries(run: Path) -> list[tuple[str, str]]:
    cfg = _json(run / "config.json")
    if cfg.get("task") != "asr":
        return []
    family = _family(cfg.get("base_model", ""), cfg.get("backend", ""))
    train = ((cfg.get("datasets") or {}).get("train") or {}).get("name", "?")
    src = (cfg.get("init_from") or {}).get("source_run")
    desc = f"FT · {family} · {run.name} · {train}" + (f" (from {src})" if src else "")
    status = _json(run / "status.json")
    msg = status.get("message", "").lower()
    if status.get("phase") != "export" and any(w in msg for w in ("error", "killed", "failed")):
        desc += " · run failed"
    wer = (_json(run / "eval" / "results.json").get("metrics") or {}).get("wer")
    out = []
    for art in _json(run / "export" / "manifest.json").get("artifacts", []):
        p = run / "export" / art.get("path", "")
        if p.is_dir():
            score = f", eval WER {wer:.1%}" if isinstance(wer, (int, float)) else ""
            out.append((f"{desc} · {art.get('format')} export{score}", str(p)))
    final = run / "checkpoints" / "final"
    if not out and ((final / "config.json").exists() or (final / "model.nemo").exists()):  # no export: final
        out.append((f"{desc} · final checkpoint", str(final)))
    return out


def registry() -> dict[str, str]:
    """{dropdown label: checkpoint}. Base models first, then runs newest first."""
    reg = dict(BASE_MODELS)
    runs = sorted((r for r in RUNS_ROOT.glob("r-*") if r.is_dir()), key=lambda r: r.stat().st_mtime, reverse=True)
    for run in runs:
        reg.update(_run_entries(run))
    return reg


def label_for(ckpt: str, reg: dict[str, str] | None = None) -> str:
    """The registry label of `ckpt`, or the checkpoint itself when it is not in the registry."""
    ckpt = ckpt.strip().rstrip("/")
    for name, path in (reg or registry()).items():
        if path.rstrip("/") == ckpt:
            return name
    return ckpt


def base_model_of(ckpt_dir: Path) -> str | None:
    """The base model a training checkpoint was fine-tuned from (its init_meta.json, else the run's config.json)."""
    for meta in (ckpt_dir / "init_meta.json", ckpt_dir.parent.parent / "config.json"):
        base = _json(meta).get("base_model")
        if base:
            return base
    return None
