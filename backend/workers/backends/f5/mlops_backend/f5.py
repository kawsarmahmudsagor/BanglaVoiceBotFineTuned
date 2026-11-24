"""F5 family handlers (IndicF5): prepare here; train/eval/export live in tts.py and are re-exported below.
huggingface_hub is imported inside functions only."""
from __future__ import annotations

from pathlib import Path

from app.core.backend_kit import max_skip_share
from app.core.contract import RunConfig, RunDir, write_prepared_refs
from app.core.prepare import PrepareError, ensure_prepared

from .models import MODELS
from .prepare.f5 import build_f5, f5_spec, pick_vocab_file, read_vocab, vocab_hash


def hub_vocab(repo: str) -> list[str]:
    """The base vocabulary: the repo's vocab file, downloaded into HF_HOME."""
    import os

    from huggingface_hub import hf_hub_download, list_repo_files

    name = pick_vocab_file(list_repo_files(repo))
    return read_vocab(Path(hf_hub_download(repo, name, cache_dir=os.environ.get("HF_HOME") or None)))


def base_vocab(rd: RunDir, cfg: RunConfig, download=None) -> list[str]:
    """Warm start (`init_from`): the init checkpoint's vocab.txt (read-only); else the base model's."""
    if cfg.init_from is not None:
        p = rd.init_dir / "vocab.txt"
        if not p.is_file():
            raise PrepareError("init_from checkpoint has no vocab.txt")
        return read_vocab(p)
    return (download or (lambda: hub_vocab(cfg.base_model)))()


def check_f5_refs(info) -> None:
    """Cache-hit safety: the first and last row of each raw.arrow must still point at existing audio."""
    import pyarrow as pa

    from app.core.backend_kit.audio_refs import check_audio_refs

    for name in info.files:
        if name.endswith("raw.arrow"):
            with pa.OSFile(str(Path(info.path) / name), "rb") as f:
                col = pa.ipc.open_stream(f).read_all().column("audio_path")
            if len(col):
                check_audio_refs([col[0].as_py(), col[len(col) - 1].as_py()], str(info.path))


def prepare_f5(rd: RunDir, cfg: RunConfig, rep) -> int:
    if MODELS[cfg.base_model].gated:
        from app.core.backend_kit.hf_access import check_hf_access

        check_hf_access(cfg.base_model)
    vocab = base_vocab(rd, cfg)
    spec = f5_spec(cfg, vocab_hash(vocab))
    share = max_skip_share(cfg)
    refs = []
    for role, key, splits in (("train", "train", ["train", "val"]), ("eval", "eval", None)):
        ds = cfg.datasets.get(key)
        if ds is None:
            continue
        pi = ensure_prepared(cfg, ds, spec, build_f5(vocab, rep), splits=splits, max_skip_share=share)
        check_f5_refs(pi)
        refs.append({"role": role, **pi.model_dump()})
    write_prepared_refs(rd, refs)
    return 0


prepare = prepare_f5  # phase handler looked up by phases._dispatch

from .tts import eval_f5 as eval  # noqa: E402,A001,F401  (phase handler names looked up by phases._dispatch)
from .tts import export_f5 as export  # noqa: E402,F401
from .tts import train  # noqa: E402,F401
