"""NeMo ASR family handlers (`nemo-manifest`). Only `prepare` so far; train/eval/export are added later.
nemo and huggingface_hub are imported inside functions only."""
from __future__ import annotations

import os
from pathlib import Path

from app.core.backend_kit import max_skip_share
from app.core.contract import RunConfig, RunDir, write_prepared_refs
from app.core.prepare import ensure_prepared

from .models import MODELS
from .prepare.manifest import build_manifest, manifest_spec, tokenizer_hash


def restore_model(base_model: str):
    """Load the model on CPU. A local .nemo path is restored directly; a Hugging Face repo holding a .nemo file
    is downloaded into HF_HOME first; anything else goes through from_pretrained."""
    from nemo.collections.asr.models import ASRModel

    if Path(base_model).is_file():
        return ASRModel.restore_from(base_model, map_location="cpu")
    try:
        from huggingface_hub import snapshot_download

        local = Path(snapshot_download(base_model, allow_patterns=["*.nemo"],
                                       cache_dir=os.environ.get("HF_HOME") or None))
        found = sorted(local.glob("*.nemo"))
        if found:
            return ASRModel.restore_from(str(found[0]), map_location="cpu")
    except Exception:  # noqa: BLE001 - not an HF repo with a .nemo file: NGC / from_pretrained below
        pass
    return ASRModel.from_pretrained(base_model, map_location="cpu")


def tokenizer_identity(tok) -> str:
    vocab = getattr(tok, "vocab", None)
    if vocab is None:
        n = int(getattr(tok, "vocab_size", 0))
        vocab = [tok.ids_to_tokens([i]) for i in range(n)] if hasattr(tok, "ids_to_tokens") else n
    return tokenizer_hash(list(vocab) if not isinstance(vocab, (int, dict)) else vocab)


def prepare(rd: RunDir, cfg: RunConfig, rep) -> int:
    if MODELS[cfg.base_model].gated:
        from app.core.backend_kit.hf_access import check_hf_access

        check_hf_access(cfg.base_model)
    model = restore_model(cfg.base_model)
    tok = model.tokenizer
    unk_id = getattr(tok, "unk_id", None)
    sha = tokenizer_identity(tok)
    del model
    spec = manifest_spec(cfg, sha)
    share = max_skip_share(cfg)
    refs = []
    for role, key, splits in (("train", "train", ["train", "val"]), ("eval", "eval", None)):
        ds = cfg.datasets.get(key)
        if ds is None:
            continue
        # The eval set keeps rows with unknown characters (they only cost WER).
        build = build_manifest(tok.text_to_ids, unk_id if role == "train" else None, rep)
        pi = ensure_prepared(cfg, ds, spec, build, splits=splits, max_skip_share=share)
        refs.append({"role": role, **pi.model_dump()})
    write_prepared_refs(rd, refs)
    return 0
