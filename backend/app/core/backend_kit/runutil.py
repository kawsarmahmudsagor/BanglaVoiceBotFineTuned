"""Run-level helpers shared by the training backends (step counts, eval row selection, trainer args).

Light: torch is imported inside free_gpu only. The package __init__ does not import this module.
"""
from __future__ import annotations

import math
from pathlib import Path

from app.core.backend_kit import StopFlag, keep_checkpoints
from app.core.contract import DatasetRef, ExportArtifact, RunConfig, RunDir, read_prepared_refs
from app.core.prepare import iter_rows


def train_ref(rd: RunDir) -> dict:
    for r in read_prepared_refs(rd):
        if r.get("role") == "train":
            return r
    raise RuntimeError("no prepared train dataset (prepared_refs.json has no train entry)")


def total_steps(cfg: RunConfig, rd: RunDir) -> int:
    hp = cfg.hparams
    if hp.get("max_steps"):
        return int(hp["max_steps"])
    n = int(train_ref(rd).get("counts", {}).get("train", 0))
    per_step = int(hp.get("batch_size", 2)) * int(hp.get("grad_accum", 4))
    return max(1, math.ceil(n / per_step) * math.ceil(float(hp.get("epochs", 1))))


def trainer_args(cfg: RunConfig, rd: RunDir, optim: str) -> dict:
    """TrainingArguments kwargs as a plain dict (max_steps overrides num_train_epochs)."""
    hp = cfg.hparams
    save_steps = max(50, total_steps(cfg, rd) // 10)
    args: dict = {
        "output_dir": str(rd.checkpoints_dir),
        "per_device_train_batch_size": int(hp.get("batch_size", 2)),
        "per_device_eval_batch_size": int(hp.get("batch_size", 2)),
        "gradient_accumulation_steps": int(hp.get("grad_accum", 4)),
        "learning_rate": float(hp.get("learning_rate", 2e-4)),
        "warmup_steps": int(hp.get("warmup_steps", 5)),
        "save_steps": save_steps,
        "save_total_limit": keep_checkpoints(hp),
        "logging_steps": 10,
        "eval_strategy": "steps",
        "eval_steps": save_steps,
        "bf16": True,
        "optim": optim,
        "report_to": [],
        "seed": cfg.seed,
        "remove_unused_columns": False,
    }
    if hp.get("max_steps"):
        args["max_steps"] = int(hp["max_steps"])
    else:
        args["num_train_epochs"] = float(hp.get("epochs", 1))
    return args


def stopped_early(stop: StopFlag, trainer) -> bool:
    """SIGTERM arrived before an HF Trainer reached max_steps: the callback saved a checkpoint, exit 143.
    A stop that lands after training finished naturally is not early: the caller writes `final` and exits 0."""
    if not stop.requested:
        return False
    state = trainer.state
    max_steps = int(getattr(state, "max_steps", 0) or 0)
    return max_steps <= 0 or int(state.global_step) < max_steps


def eval_ds(cfg: RunConfig) -> tuple[DatasetRef, list[str]]:
    """Dataset and splits used for eval: the eval dataset's own splits, else the train dataset's `val`."""
    ev = cfg.datasets.get("eval")
    if ev is not None:
        return ev, list(ev.splits)
    return cfg.datasets["train"], ["val"]


def eval_rows(cfg: RunConfig) -> list[dict]:
    ds, splits = eval_ds(cfg)
    return [row for s in splits for row in iter_rows(ds, s)]


def eval_loss_files(rd: RunDir, cfg: RunConfig) -> list[Path]:
    """Prepared files behind the eval rows (same selection as eval_rows): `<split>.jsonl` per split
    (whisper uses the `<split>` dir with the suffix dropped)."""
    role = "eval" if cfg.datasets.get("eval") is not None else "train"
    ref = next((r for r in read_prepared_refs(rd) if r.get("role") == role), None)
    if ref is None:
        raise RuntimeError(f"no prepared {role} dataset (prepared_refs.json has no {role} entry)")
    _, splits = eval_ds(cfg)
    return [Path(ref["path"]) / f"{s}.jsonl" for s in splits]


def free_gpu() -> None:
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()


def dir_artifact(rd: RunDir, fmt: str, d: Path, serve_with: str) -> ExportArtifact:
    size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    return ExportArtifact(format=fmt, path=d.relative_to(rd.export_dir).as_posix(),
                          size_bytes=size, serve_with=serve_with)
