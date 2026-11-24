"""Import shims for upstream CosyVoice.

`cosyvoice.utils.train_utils` imports deepspeed at module level (`import deepspeed` and
`from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live`); at call time
deepspeed is only used when `train_engine == "deepspeed"`, which this backend never selects (torch_ddp only). When
deepspeed is not installed we register a stub whose callables raise, instead of making it a dependency.
`cosyvoice.utils.executor` imports train_utils, so call this before importing either.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types


def _unsupported(*_a, **_k):
    raise RuntimeError("deepspeed engine not supported")


def ensure_deepspeed_stub() -> None:
    if "deepspeed" in sys.modules or importlib.util.find_spec("deepspeed") is not None:
        return
    names = ["deepspeed", "deepspeed.runtime", "deepspeed.runtime.zero", "deepspeed.runtime.zero.stage_1_and_2"]
    mods = {n: types.ModuleType(n) for n in names}
    for n, m in mods.items():  # find_spec raises on a sys.modules entry whose __spec__ is None (transformers probes it)
        m.__spec__ = importlib.machinery.ModuleSpec(n, None, is_package=n != names[-1])
    for parent, child in zip(names, names[1:]):
        setattr(mods[parent], child.rsplit(".", 1)[1], mods[child])
    for fn in ("init_distributed", "initialize", "add_config_arguments"):
        setattr(mods["deepspeed"], fn, _unsupported)
    mods["deepspeed.runtime.zero.stage_1_and_2"].estimate_zero2_model_states_mem_needs_all_live = _unsupported
    sys.modules.update(mods)


def _nvidia_lib_dirs() -> list:
    from pathlib import Path

    dirs = []
    for entry in sys.path:
        root = Path(entry) / "nvidia"
        if root.is_dir():
            dirs.extend(sorted(p for p in root.glob("*/lib") if p.is_dir()))
    return dirs


def preload_cuda_libs(lib_dirs=None) -> int:
    """Load the CUDA/cuDNN libraries shipped in torch's nvidia-* wheels into the process (RTLD_GLOBAL), so the
    onnxruntime-gpu 1.20 CUDA provider finds e.g. libcudnn_adv.so.9 without LD_LIBRARY_PATH (onnxruntime 1.21+
    has preload_dlls for this). Linux only; returns how many libraries loaded. Unloadable ones are skipped."""
    import ctypes

    if not sys.platform.startswith("linux"):
        return 0
    libs = [f for d in (lib_dirs if lib_dirs is not None else _nvidia_lib_dirs()) for f in sorted(d.glob("lib*.so*"))]
    loaded, pending = 0, libs
    for _ in range(3):  # a few passes so a library whose dependency loads later still gets in
        failed = []
        for f in pending:
            try:
                ctypes.CDLL(str(f), mode=ctypes.RTLD_GLOBAL)
                loaded += 1
            except OSError:
                failed.append(f)
        if not failed or len(failed) == len(pending):
            break
        pending = failed
    return loaded

