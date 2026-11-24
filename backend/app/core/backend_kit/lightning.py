"""PyTorch Lightning glue for the contract (NeMo backend). Not imported by the backend_kit package __init__.

`lightning` is only installed in the nemo venv; without it the base classes fall back to `object` so the pure
logic stays testable in the light workspace venv.

The nemo backend targets single-node training; multi-node needs a shared filesystem. Metric, status and
checkpoint-marker writes happen on global rank 0 only.
"""
from __future__ import annotations

from pathlib import Path

from app.core.backend_kit import NanGuard, Reporter, StopFlag, mark_complete, write_init_meta
from app.core.contract import Phase, RunDir

try:  # pragma: no cover - exercised on the VM
    from lightning.pytorch import Callback
    from lightning.pytorch.callbacks import ModelCheckpoint
except ImportError:  # light venv
    Callback = object  # type: ignore[assignment,misc]
    ModelCheckpoint = object  # type: ignore[assignment,misc]

CKPT_NAME = "model.ckpt"
CKPT_FORMAT = "lightning-ckpt"  # spec §14.4


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _rank0(trainer) -> bool:
    return bool(getattr(trainer, "is_global_zero", True))


def _optimizer_boundary(trainer, batch_idx: int) -> bool:
    """True when this micro-batch completed an optimizer step (or ended the epoch)."""
    accum = max(int(getattr(trainer, "accumulate_grad_batches", 1) or 1), 1)
    last = getattr(trainer, "num_training_batches", None)
    if (batch_idx + 1) % accum == 0:
        return True
    return isinstance(last, int) and batch_idx + 1 >= last


def finalize_checkpoint(ckpt_path: Path, *, base_model: str, step: int) -> None:
    """After the .ckpt is fully written: init_meta first, then `.complete`."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not written: {ckpt_path}")
    write_init_meta(ckpt_path.parent, backend="nemo", base_model=base_model, method="full",
                    format=CKPT_FORMAT, step=step)
    mark_complete(ckpt_path.parent)


class ContractCheckpoint(ModelCheckpoint):
    """Saves `checkpoints/step-<n>/model.ckpt` and marks it complete only after Lightning wrote the file.

    (Callback.on_save_checkpoint fires while the checkpoint dict is built, before the file exists.)
    """

    def __init__(self, rd: RunDir, base_model: str, every_n_train_steps: int | None = None):
        if ModelCheckpoint is not object:
            super().__init__(dirpath=str(rd.checkpoints_dir), filename="step-{step}/" + Path(CKPT_NAME).stem,
                             save_top_k=-1, every_n_train_steps=every_n_train_steps, save_on_train_epoch_end=False,
                             auto_insert_metric_name=False,
                             enable_version_counter=False)  # re-save at a step never becomes model-v1.ckpt
        self.rd = rd
        self.base_model = base_model

    def _save_checkpoint(self, trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        finalize_checkpoint(Path(filepath), base_model=self.base_model, step=int(trainer.global_step))

    def save_now(self, trainer) -> Path:
        path = self.rd.checkpoints_dir / f"step-{int(trainer.global_step)}" / CKPT_NAME
        if _rank0(trainer):
            self._save_checkpoint(trainer, str(path))
        return path


class ContractLightningCallback(Callback):
    def __init__(self, rep: Reporter, stop: StopFlag, nan: NanGuard, total_steps: int):
        self.rep = rep
        self.stop = stop
        self.nan = nan
        self.total_steps = total_steps
        self._stop_saved = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not _optimizer_boundary(trainer, batch_idx):
            return
        step = int(trainer.global_step)
        loss = _num(trainer.callback_metrics.get("train_loss"))
        if loss is not None:
            if _rank0(trainer):
                self.rep.metric(step, "train", "loss", loss)
            self.nan.feed(loss)  # raises TrainingDiverged after 3 non-finite in a row
        if _rank0(trainer):
            self.rep.status(Phase.train, step, self.total_steps, "training")
        if self.stop.requested:
            trainer.should_stop = True
            if not self._stop_saved:
                self._stop_saved = True
                for cb in getattr(trainer, "checkpoint_callbacks", []):
                    if isinstance(cb, ContractCheckpoint):
                        cb.save_now(trainer)
                        break

    def on_validation_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False) or not _rank0(trainer):
            return
        step = int(trainer.global_step)
        for key, name in (("val_loss", "loss"), ("val_wer", "wer")):
            v = _num(trainer.callback_metrics.get(key))
            if v is not None:
                self.rep.metric(step, "val", name, v)
