"""HF Trainer glue for the contract. Imports transformers: only HF-based backends import this module."""
from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

from transformers import TrainerCallback

from app.core.backend_kit import NanGuard, Reporter, StopFlag, mark_complete
from app.core.contract import Phase

_TRAIN_KEYS = ("loss", "learning_rate", "grad_norm")


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


class ContractCallback(TrainerCallback):
    def __init__(self, rep: Reporter, stop: StopFlag, nan: NanGuard, total_steps: int):
        self.rep = rep
        self.stop = stop
        self.nan = nan
        self.total_steps = total_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        for k, v in (logs or {}).items():
            if k in _TRAIN_KEYS and _is_num(v):
                self.rep.metric(state.global_step, "train", k, float(v))
            elif k == "eval_loss" and _is_num(v):
                self.rep.metric(state.global_step, "val", "loss", float(v))
        loss = (logs or {}).get("loss")
        if _is_num(loss):
            self.nan.feed(float(loss))  # raises TrainingDiverged after 3 non-finite in a row

    def on_step_end(self, args, state, control, **kwargs):
        self.rep.status(Phase.train, state.global_step, self.total_steps, "training")
        if self.stop.requested:
            control.should_save = True
            control.should_training_stop = True

    def on_save(self, args, state, control, **kwargs):
        mark_complete(Path(args.output_dir) / f"checkpoint-{state.global_step}")


class MetaContractCallback(ContractCallback):
    """ContractCallback that writes init_meta into each checkpoint BEFORE `.complete` appears."""

    def __init__(self, rep, stop, nan, total_steps: int, write_meta: Callable[[Path, int], None]):
        super().__init__(rep, stop, nan, total_steps)
        self.write_meta = write_meta

    def on_save(self, args, state, control, **kwargs):
        self.write_meta(Path(args.output_dir) / f"checkpoint-{state.global_step}", state.global_step)
        super().on_save(args, state, control, **kwargs)
