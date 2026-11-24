"""NeMo ASR family handlers (`nemo-manifest`). Only `prepare` so far; train/eval/export are added later.
nemo and huggingface_hub are imported inside functions only."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from app.core.backend_kit import max_skip_share
from app.core.contract import RunConfig, RunDir, write_prepared_refs
from app.core.backend_kit.audio_refs import check_audio_refs, first_last_lines
from app.core.prepare import PrepareError, ensure_prepared

from .models import MODELS
from .prepare.manifest import build_manifest, manifest_spec, tokenizer_hash


def restore_model(base_model: str):
    """Load the model on CPU. A local .nemo path is restored directly; a Hugging Face repo holding a .nemo file
    is downloaded into HF_HOME first; anything else goes through from_pretrained."""
    from nemo.collections.asr.models import ASRModel

    if Path(base_model).is_file():
        return ASRModel.restore_from(base_model, map_location="cpu")
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    try:
        local = Path(snapshot_download(base_model, allow_patterns=["*.nemo"],
                                       cache_dir=os.environ.get("HF_HOME") or None))
    except (RepositoryNotFoundError, HfHubHTTPError, ValueError) as e:  # not an HF repo: NGC / from_pretrained
        print(f"nemo: no .nemo download for {base_model} ({type(e).__name__}); trying from_pretrained", file=sys.stderr)
        local = None
    found = sorted(local.glob("*.nemo")) if local else []
    if found:
        return ASRModel.restore_from(str(found[0]), map_location="cpu")
    return ASRModel.from_pretrained(base_model, map_location="cpu")


def tokenizer_identity(tok) -> str:
    """Hash of the tokenizer vocabulary (a model change re-prepares)."""
    vocab = getattr(tok, "vocab", None)
    if vocab is None:
        vocab = [tok.ids_to_tokens([i]) for i in range(int(tok.vocab_size))]
    return tokenizer_hash(vocab if isinstance(vocab, dict) else list(vocab))


def resolve_unk_id(tok, model_name: str) -> int:
    unk = getattr(tok, "unk_id", None)
    if unk is None and hasattr(tok, "token_to_id"):
        unk = tok.token_to_id("<unk>")
    if unk is None and hasattr(tok, "tokens_to_ids"):
        try:
            ids = tok.tokens_to_ids(["<unk>"])
            unk = ids[0] if ids else None
        except Exception:  # noqa: BLE001 - tokenizer without that token
            unk = None
    if unk is None:
        raise PrepareError(f"tokenizer of {model_name} has no unk id; cannot guard unknown characters")
    return int(unk)


def check_manifest_refs(info) -> None:
    """Cache-hit safety: the first and last row of each manifest must still point at existing audio."""
    for name in info.files:
        if name.endswith(".jsonl"):
            ends = first_last_lines(Path(info.path) / name)
            check_audio_refs([r["audio_filepath"] for r in ends], str(info.path))


def prepare_nemo(rd: RunDir, cfg: RunConfig, rep) -> int:
    if MODELS[cfg.base_model].gated:
        from app.core.backend_kit.hf_access import check_hf_access

        check_hf_access(cfg.base_model)
    model = restore_model(cfg.base_model)
    tok = model.tokenizer
    unk_id = resolve_unk_id(tok, cfg.base_model)
    sha = tokenizer_identity(tok)
    del model
    spec = manifest_spec(cfg, sha)
    share = max_skip_share(cfg)
    refs = []
    for role, key, splits in (("train", "train", ["train", "val"]), ("eval", "eval", None)):
        ds = cfg.datasets.get(key)
        if ds is None:
            continue
        build = build_manifest(tok.text_to_ids, unk_id, rep)
        pi = ensure_prepared(cfg, ds, spec, build, splits=splits, max_skip_share=share)
        check_manifest_refs(pi)
        refs.append({"role": role, **pi.model_dump()})
    write_prepared_refs(rd, refs)
    return 0


prepare = prepare_nemo  # phase handler looked up by phases._dispatch
