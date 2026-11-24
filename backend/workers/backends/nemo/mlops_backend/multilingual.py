"""Load AI4Bharat's multilingual IndicConformer .nemo as a stock, single-language NeMo model.

The `ai4bharat/indicconformer_stt_*` checkpoints were saved by AI4Bharat's NeMo fork (1.23): the tokenizer is
`type: multilingual` (one 256-token SentencePiece model per Indic language) and the decoder, joint and CTC heads are
"multisoftmax". Stock NeMo cannot build that config. Since every language shares the encoder, we keep the encoder,
take the target language's tokenizer, and cut the heads down to that language:

- decoder.prediction.embed.weight (N*V + 1, H): rows [i*V, (i+1)*V) of language i, plus the final blank row
- joint.joint_net.2.<lang>.* (V + 1, H): the language's own head becomes joint.joint_net.2.*
- CTC: a per-language layer (`.<lang>.` in the key) is renamed the same way; a shared (N*V + 1)-row layer is sliced

The result is a plain EncDecHybridRNNTCTCBPEModel that trains, evaluates and saves (.nemo) with stock NeMo 3.0.
Pure config/state-dict helpers are importable without torch or nemo (the tests use numpy arrays).
"""
from __future__ import annotations

import copy
import shutil
import tarfile
import tempfile
from pathlib import Path

LANG = "bn"
HEAD_PREFIXES = ("joint.", "ctc_decoder.")
# keys the fork adds that stock NeMo rejects or misreads
_FORK_KEYS = {"decoder": ("multisoftmax",), "joint": ("multilingual", "language_keys"),
              "aux_ctc.decoder": ("multisoftmax",)}


def multilingual_langs(cfg: dict) -> list[str] | None:
    """Language order of a multilingual tokenizer config (it fixes each language's token offset), else None."""
    tok = cfg.get("tokenizer") or {}
    if tok.get("type") != "multilingual":
        return None
    return list((tok.get("langs") or {}).keys())


def _drop(d: dict, path: str, keys) -> None:
    node = d
    for part in path.split("."):
        node = node.get(part) if isinstance(node, dict) else None
        if node is None:
            return
    for k in keys:
        node.pop(k, None)


def monolingual_config(cfg: dict, lang: str, tokenizer_dir: str) -> dict:
    """The fork's config rewritten for one language: a plain BPE tokenizer in `tokenizer_dir`, no fork-only keys,
    no language-id outputs, and no dataset paths from the authors' cluster. Vocabulary sizes and lists are
    recomputed by the model from the tokenizer."""
    out = copy.deepcopy(cfg)
    if lang not in (multilingual_langs(cfg) or []):
        raise ValueError(f"language {lang!r} not in the multilingual tokenizer")
    out["tokenizer"] = {"dir": str(tokenizer_dir), "type": "bpe"}
    for path, keys in _FORK_KEYS.items():
        _drop(out, path, keys)
    for ds in ("train_ds", "validation_ds", "test_ds"):
        if isinstance(out.get(ds), dict):
            out[ds].pop("return_language_id", None)
            out[ds]["manifest_filepath"] = None
            out[ds].pop("tarred_audio_filepaths", None)
    out.pop("labels", None)
    return out


def _segments(key: str) -> list[str]:
    return key.split(".")


def _concat(a, b):
    if hasattr(a, "new_tensor"):  # torch
        import torch

        return torch.cat([a, b], dim=0)
    import numpy as np

    return np.concatenate([a, b], axis=0)


def _slice_rows(w, idx: int, vocab: int):
    return _concat(w[idx * vocab:(idx + 1) * vocab], w[-1:])


def monolingual_state_dict(sd: dict, langs: list[str], lang: str) -> dict:
    """State dict of the multilingual model -> state dict of the single-language model (see module docstring)."""
    idx = langs.index(lang)
    others = set(langs) - {lang}
    emb = sd["decoder.prediction.embed.weight"]
    total = emb.shape[0] - 1
    if total % len(langs):
        raise ValueError(f"embedding rows {emb.shape[0]} do not split into {len(langs)} languages + blank")
    vocab = total // len(langs)
    out = {}
    for key, w in sd.items():
        if key == "decoder.prediction.embed.weight":
            out[key] = _slice_rows(w, idx, vocab)
            continue
        if not key.startswith(HEAD_PREFIXES):
            out[key] = w
            continue
        segs = _segments(key)
        if others & set(segs):
            continue  # another language's head
        if lang in segs:
            out[".".join(s for s in segs if s != lang)] = w
        elif getattr(w, "shape", None) and w.shape[0] == total + 1:
            out[key] = _slice_rows(w, idx, vocab)  # a shared head over every language's tokens
        else:
            out[key] = w
    return out


def _member(tar: tarfile.TarFile, name: str) -> tarfile.TarInfo:
    for cand in (name, f"./{name}"):
        try:
            return tar.getmember(cand)
        except KeyError:
            continue
    raise KeyError(f"{name} not in the .nemo archive")


def _extract(tar: tarfile.TarFile, name: str, dest: Path) -> Path:
    src = tar.extractfile(_member(tar, name))
    with dest.open("wb") as f:
        shutil.copyfileobj(src, f)
    return dest


def read_nemo_config(path: str | Path) -> dict:
    import yaml

    with tarfile.open(path) as tar:
        return yaml.safe_load(tar.extractfile(_member(tar, "model_config.yaml")))


def restore_monolingual(path: str | Path, lang: str = LANG, cfg: dict | None = None):
    """Build the single-language EncDecHybridRNNTCTCBPEModel from a multilingual IndicConformer .nemo (on CPU).
    The tokenizer files go to a private temp dir that lives as long as the process (NeMo reads it again on save)."""
    import torch
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
    from omegaconf import OmegaConf

    cfg = cfg if cfg is not None else read_nemo_config(path)
    langs = multilingual_langs(cfg)
    if not langs:
        raise ValueError(f"{path} is not a multilingual IndicConformer checkpoint")
    entry = cfg["tokenizer"]["langs"][lang]
    tok_dir = Path(tempfile.mkdtemp(prefix=f"indicconformer-{lang}-tok-"))
    work = Path(tempfile.mkdtemp(prefix="indicconformer-w-"))
    try:
        with tarfile.open(path) as tar:
            for field, name in (("model_path", "tokenizer.model"), ("vocab_path", "vocab.txt"),
                                ("spe_tokenizer_vocab", "tokenizer.vocab")):
                ref = str(entry.get(field) or "")
                if ref:
                    _extract(tar, ref.removeprefix("nemo:"), tok_dir / name)
            weights = _extract(tar, "model_weights.ckpt", work / "model_weights.ckpt")
        sd = torch.load(weights, map_location="cpu", weights_only=False)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    model = EncDecHybridRNNTCTCBPEModel(cfg=OmegaConf.create(monolingual_config(cfg, lang, str(tok_dir))))
    mono = monolingual_state_dict(sd, langs, lang)
    missing, unexpected = model.load_state_dict(mono, strict=False)
    missing = [k for k in missing if not k.startswith("preprocessor.")]  # feature buffers are rebuilt
    if missing or unexpected:
        raise RuntimeError(f"IndicConformer {lang} conversion: missing {missing[:10]}, unexpected {unexpected[:10]}"
                           f" (heads in checkpoint: {[k for k in sd if k.startswith(HEAD_PREFIXES)][:12]})")
    return model.eval()
