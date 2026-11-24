"""VM install check: `python -m mlops_backend.check`. Exits 1 if CUDA or an import is missing."""
from __future__ import annotations

import sys
from importlib import import_module, metadata

_DIST = {"f5_tts": "f5-tts"}


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
    for mod in ["torchaudio", "f5_tts", "accelerate"]:
        try:
            import_module(mod)
            print(f"{mod} {metadata.version(_DIST.get(mod, mod))}")
        except Exception as e:  # noqa: BLE001
            print(f"{mod} import failed: {type(e).__name__}: {e}", file=sys.stderr)
            ok = False
    for mod in ["f5_tts.model"]:
        try:
            import_module(mod)
            print(f"{mod} OK")
        except Exception as e:  # noqa: BLE001
            print(f"{mod} import failed: {type(e).__name__}: {e}", file=sys.stderr)
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
