"""VM install check: `python -m mlops_backend.check`. Exits 1 if CUDA or an import is missing."""
from __future__ import annotations

import os
import sys
from importlib import import_module, metadata
from pathlib import Path

_DIST = {"whisper": "openai-whisper", "hyperpyyaml": "HyperPyYAML", "x_transformers": "x-transformers",
         "onnxruntime": "onnxruntime-gpu"}
_REPO_DIR = "models--FunAudioLLM--Fun-CosyVoice3-0.5B-2512"
_BACKEND_DIR = Path(__file__).resolve().parents[1]


def _upstream_commit() -> str:
    try:
        return (_BACKEND_DIR / "UPSTREAM_COMMIT").read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError):
        return "unknown"


def _model_dir() -> Path | None:
    """Local snapshot of the base model in the hub cache (HF_HUB_CACHE, else $HF_HOME/hub, where
    download_model puts it), or None when not downloaded yet."""
    root = os.environ.get("HF_HUB_CACHE")
    if not root:
        hf_home = os.environ.get("HF_HOME")
        if not hf_home:
            return None
        root = str(Path(hf_home) / "hub")
    snaps = sorted((Path(root) / _REPO_DIR / "snapshots").glob("*/cosyvoice3.yaml"))
    return snaps[-1].parent if snaps else None


def _load_config() -> bool:
    """Instantiate cosyvoice3.yaml without loading weights. Returns False on failure."""
    model_dir = _model_dir()
    if model_dir is None:
        print("cosyvoice3.yaml: skipped (model not downloaded under HF_HOME yet)")
        return True
    try:
        from hyperpyyaml import load_hyperpyyaml

        with open(model_dir / "cosyvoice3.yaml", encoding="utf-8") as f:
            cfg = load_hyperpyyaml(f, overrides={"qwen_pretrain_path": str(model_dir / "CosyVoice-BlankEN")})
        print(f"cosyvoice3.yaml OK ({', '.join(k for k in ('llm', 'flow', 'hift') if k in cfg)})")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"cosyvoice3.yaml load failed: {type(e).__name__}: {e}", file=sys.stderr)
        return False


def main() -> int:
    ok = True
    try:
        import torch

        cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
        print(f"torch {torch.__version__} cuda {torch.cuda.is_available()} "
              f"capability {f'{cap[0]}.{cap[1]}' if cap else 'n/a'}")
        ok = torch.cuda.is_available()
    except Exception as e:  # noqa: BLE001
        print(f"torch import failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    if not torch.__version__.startswith("2.7.1"):
        print(f"warning: expected torch 2.7.1+cu128, got {torch.__version__}", file=sys.stderr)
    print(f"upstream commit {_upstream_commit()}")
    for mod in ["torchaudio", "onnxruntime", "hyperpyyaml", "conformer", "diffusers", "lightning",
                "x_transformers", "transformers", "librosa", "whisper"]:
        try:
            import_module(mod)
            print(f"{mod} {metadata.version(_DIST.get(mod, mod))}")
        except Exception as e:  # noqa: BLE001
            print(f"{mod} import failed: {type(e).__name__}: {e}", file=sys.stderr)
            ok = False
    try:
        import onnxruntime

        providers = onnxruntime.get_available_providers()
        print(f"onnxruntime providers {providers}")
        if "CUDAExecutionProvider" not in providers:
            print("onnxruntime CUDAExecutionProvider missing", file=sys.stderr)
            ok = False
    except Exception:  # noqa: BLE001 - already reported by the import loop
        pass
    for mod in ["cosyvoice", "matcha"]:
        try:
            import_module(mod)
            print(f"{mod} OK")
        except Exception as e:  # noqa: BLE001
            print(f"{mod} import failed: {type(e).__name__}: {e}", file=sys.stderr)
            ok = False
    if not _load_config():
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
