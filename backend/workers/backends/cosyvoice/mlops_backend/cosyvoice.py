"""CosyVoice family handlers (Fun-CosyVoice3): prepare here; train/eval/export live in train.py / eval.py / export.py
and are re-exported below.
huggingface_hub is imported inside functions only."""
from __future__ import annotations

from pathlib import Path

from app.core.backend_kit import max_skip_share
from app.core.contract import RunConfig, RunDir, write_prepared_refs
from app.core.prepare import ensure_prepared

from .models import MODELS
from .prepare.cosyvoice import build_cosyvoice, check_cosyvoice_refs, cosyvoice_spec

# Files of the hub repo the backend uses (skips llm.rl.pt, the batch tokenizer and the fp32 estimator onnx).
MODEL_PATTERNS = ["llm.pt", "flow.pt", "hift.pt", "speech_tokenizer_v3.onnx", "campplus.onnx", "cosyvoice3.yaml",
                  "CosyVoice-BlankEN/*"]


def download_model(repo: str) -> Path:
    """The model snapshot directory in the default hub cache ($HF_HOME/hub); its basename is the revision."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, allow_patterns=MODEL_PATTERNS))


def prepare_cosyvoice(rd: RunDir, cfg: RunConfig, rep, download=None) -> int:
    if MODELS[cfg.base_model].gated:
        from app.core.backend_kit.hf_access import check_hf_access

        check_hf_access(cfg.base_model)
    model_dir = (download or download_model)(cfg.base_model)
    spec = cosyvoice_spec(Path(model_dir).name)
    share = max_skip_share(cfg)
    refs = []
    for role, key, splits in (("train", "train", ["train", "val"]), ("eval", "eval", None)):
        ds = cfg.datasets.get(key)
        if ds is None:
            continue
        pi = ensure_prepared(cfg, ds, spec, build_cosyvoice(model_dir, rep), splits=splits, max_skip_share=share)
        check_cosyvoice_refs(pi)
        refs.append({"role": role, **pi.model_dump()})
    write_prepared_refs(rd, refs)
    return 0


prepare = prepare_cosyvoice  # phase handler looked up by phases._dispatch

from .eval import eval_cosyvoice as eval  # noqa: E402,A001,F401  (phase handler names looked up by phases._dispatch)
from .export import export_cosyvoice as export  # noqa: E402,F401
from .train import train_cosyvoice as train  # noqa: E402,F401
