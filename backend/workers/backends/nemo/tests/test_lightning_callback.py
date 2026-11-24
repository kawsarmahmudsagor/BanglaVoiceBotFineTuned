import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.backend_kit import NanGuard, Reporter, StopFlag, TrainingDiverged, read_init_meta
from app.core.backend_kit.lightning import ContractCheckpoint, ContractLightningCallback, finalize_checkpoint
from app.core.contract import RunDir


class FakeTrainer:
    def __init__(self, tmp_path, **metrics):
        self.callback_metrics = metrics
        self.global_step = 0
        self.should_stop = False
        self.sanity_checking = False
        self.saved: list[str] = []
        self.ckpt = ContractCheckpoint.__new__(ContractCheckpoint)  # no lightning needed
        self.ckpt.rd = RunDir(tmp_path / "run")
        self.ckpt.base_model = "x/y"
        self.checkpoint_callbacks = [self.ckpt]

    def save_checkpoint(self, filepath):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        Path(filepath).write_bytes(b"ckpt")
        self.saved.append(str(filepath))


@pytest.fixture
def rd(tmp_path):
    r = RunDir(tmp_path / "run")
    r.ensure()
    return r


def _cb(rd, stop=None):
    return ContractLightningCallback(Reporter(rd), stop or StopFlag(), NanGuard(), total_steps=10)


def _metrics(rd):
    return [l for l in rd.metrics_path.read_text(encoding="utf-8").splitlines()] if rd.metrics_path.exists() else []


def test_finalize_writes_meta_before_complete(tmp_path):
    d = tmp_path / "checkpoints" / "step-7"
    d.mkdir(parents=True)
    (d / "model.ckpt").write_bytes(b"x")
    finalize_checkpoint(d / "model.ckpt", base_model="x/y", step=7)
    assert (d / ".complete").exists()
    meta = read_init_meta(d)
    assert meta == {"backend": "nemo", "base_model": "x/y", "method": "full", "format": "lightning-ckpt", "step": 7}


def test_finalize_requires_written_file(tmp_path):
    d = tmp_path / "step-1"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        finalize_checkpoint(d / "model.ckpt", base_model="x/y", step=1)
    assert not (d / ".complete").exists()


def test_train_batch_end_reports_train_loss(rd, tmp_path):
    tr = FakeTrainer(tmp_path, train_loss=0.5)
    tr.global_step = 3
    _cb(rd).on_train_batch_end(tr, None, None, None, 0)
    lines = _metrics(rd)
    assert len(lines) == 1 and '"train"' in lines[0] and '"loss"' in lines[0] and "0.5" in lines[0]
    assert not tr.should_stop


def test_nan_guard_raises_after_three(rd, tmp_path):
    tr = FakeTrainer(tmp_path, train_loss=math.nan)
    cb = _cb(rd)
    cb.on_train_batch_end(tr, None, None, None, 0)
    cb.on_train_batch_end(tr, None, None, None, 1)
    with pytest.raises(TrainingDiverged, match="loss is NaN; lower learning_rate"):
        cb.on_train_batch_end(tr, None, None, None, 2)


def test_stop_requests_stop_and_saves_complete_checkpoint(rd, tmp_path):
    stop = StopFlag()
    stop.requested = True
    tr = FakeTrainer(tmp_path, train_loss=0.1)
    tr.global_step = 4
    tr.ckpt._save_checkpoint = lambda trainer, fp: (trainer.save_checkpoint(fp),
                                                    finalize_checkpoint(fp, base_model="x/y", step=trainer.global_step))
    _cb(rd, stop).on_train_batch_end(tr, None, None, None, 0)
    assert tr.should_stop
    d = rd.checkpoints_dir / "step-4"
    assert (d / "model.ckpt").exists() and (d / ".complete").exists()


def test_validation_end_routes_to_val_split(rd, tmp_path):
    tr = FakeTrainer(tmp_path, val_loss=1.5, val_wer=0.4, train_loss=9.0)
    tr.global_step = 5
    _cb(rd).on_validation_end(tr, None)
    lines = _metrics(rd)
    assert len(lines) == 2 and all('"val"' in l for l in lines)
    assert any('"wer"' in l for l in lines) and any('"loss"' in l for l in lines)


def test_validation_end_skips_sanity_check(rd, tmp_path):
    tr = FakeTrainer(tmp_path, val_loss=1.5)
    tr.sanity_checking = True
    _cb(rd).on_validation_end(tr, None)
    assert _metrics(rd) == []


def test_package_init_does_not_import_lightning():
    import subprocess
    import sys
    code = "import sys, app.core.backend_kit; assert 'lightning' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_accumulation_reports_once_per_optimizer_step(rd, tmp_path):
    tr = FakeTrainer(tmp_path, train_loss=0.5)
    tr.accumulate_grad_batches = 2
    tr.num_training_batches = 10
    cb = _cb(rd)
    tr.global_step = 0
    cb.on_train_batch_end(tr, None, None, None, 0)  # micro-batch 1 of 2
    assert _metrics(rd) == []
    tr.global_step = 1
    cb.on_train_batch_end(tr, None, None, None, 1)  # optimizer step done
    assert len(_metrics(rd)) == 1 and '"step":1' in _metrics(rd)[0].replace(" ", "")


def test_accumulation_last_batch_of_epoch_counts(rd, tmp_path):
    tr = FakeTrainer(tmp_path, train_loss=0.5)
    tr.accumulate_grad_batches = 2
    tr.num_training_batches = 3
    _cb(rd).on_train_batch_end(tr, None, None, None, 2)
    assert len(_metrics(rd)) == 1


def test_stop_save_only_at_optimizer_boundary(rd, tmp_path):
    stop = StopFlag()
    stop.requested = True
    tr = FakeTrainer(tmp_path, train_loss=0.1)
    tr.accumulate_grad_batches = 2
    tr.num_training_batches = 10
    tr.ckpt._save_checkpoint = lambda trainer, fp: (trainer.save_checkpoint(fp),
                                                    finalize_checkpoint(fp, base_model="x/y", step=trainer.global_step))
    cb = _cb(rd, stop)
    cb.on_train_batch_end(tr, None, None, None, 0)
    assert not tr.saved
    tr.global_step = 1
    cb.on_train_batch_end(tr, None, None, None, 1)
    assert tr.should_stop and len(tr.saved) == 1


def test_non_zero_rank_does_nothing(rd, tmp_path):
    stop = StopFlag()
    stop.requested = True
    tr = FakeTrainer(tmp_path, train_loss=0.5, val_loss=1.0)
    tr.is_global_zero = False
    cb = _cb(rd, stop)
    cb.on_train_batch_end(tr, None, None, None, 0)
    cb.on_validation_end(tr, None)
    assert _metrics(rd) == [] and not tr.saved
    assert tr.should_stop  # every rank must still stop
