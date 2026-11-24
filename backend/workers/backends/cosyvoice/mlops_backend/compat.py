"""Import shims for upstream CosyVoice.

`cosyvoice.utils.train_utils` imports deepspeed at module level (`import deepspeed` and
`from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live`); at call time
deepspeed is only used when `train_engine == "deepspeed"`, which this backend never selects (torch_ddp only). When
deepspeed is not installed we register a stub whose callables raise, instead of making it a dependency.
`cosyvoice.utils.executor` imports train_utils, so call this before importing either.
"""
from __future__ import annotations

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
    for parent, child in zip(names, names[1:]):
        setattr(mods[parent], child.rsplit(".", 1)[1], mods[child])
    for fn in ("init_distributed", "initialize", "add_config_arguments"):
        setattr(mods["deepspeed"], fn, _unsupported)
    mods["deepspeed.runtime.zero.stage_1_and_2"].estimate_zero2_model_states_mem_needs_all_live = _unsupported
    sys.modules.update(mods)
