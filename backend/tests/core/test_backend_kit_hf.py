from types import SimpleNamespace

import pytest

pytest.importorskip("transformers")

from app.core.backend_kit import NanGuard, StopFlag, TrainingDiverged  # noqa: E402
from app.core.backend_kit.hf import ContractCallback  # noqa: E402
from app.core.contract import RunDir, read_metrics, read_status  # noqa: E402
from app.core.backend_kit import Reporter  # noqa: E402


def make(tmp_path, total=10):
    rd = RunDir(tmp_path / "run")
    rd.ensure()
    stop = StopFlag()
    return rd, stop, ContractCallback(Reporter(rd), stop, NanGuard(), total)


def test_on_log_maps_metrics_and_splits(tmp_path):
    rd, _, cb = make(tmp_path)
    state = SimpleNamespace(global_step=4)
    cb.on_log(None, state, None, logs={"loss": 1.5, "learning_rate": 1e-4, "grad_norm": 0.3,
                                       "epoch": 1.0, "eval_loss": 1.2})
    ms, _ = read_metrics(rd)
    got = {(m.split, m.name): m.value for m in ms}
    assert got == {("train", "loss"): 1.5, ("train", "learning_rate"): 1e-4,
                   ("train", "grad_norm"): 0.3, ("val", "loss"): 1.2}
    assert all(m.step == 4 for m in ms)


def test_nan_loss_three_times_diverges(tmp_path):
    _, _, cb = make(tmp_path)
    state = SimpleNamespace(global_step=1)
    cb.on_log(None, state, None, logs={"loss": float("nan")})
    cb.on_log(None, state, None, logs={"loss": float("inf")})
    with pytest.raises(TrainingDiverged, match="loss is NaN; lower learning_rate"):
        cb.on_log(None, state, None, logs={"loss": float("nan")})


def test_step_end_reports_and_stops_on_sigterm(tmp_path):
    rd, stop, cb = make(tmp_path)
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    cb.on_step_end(None, SimpleNamespace(global_step=2), control)
    assert read_status(rd).step == 2 and not control.should_training_stop
    stop.requested = True
    cb.on_step_end(None, SimpleNamespace(global_step=3), control)
    assert control.should_save and control.should_training_stop


def test_on_save_marks_checkpoint_complete(tmp_path):
    _, _, cb = make(tmp_path)
    ck = tmp_path / "out" / "checkpoint-7"
    ck.mkdir(parents=True)
    cb.on_save(SimpleNamespace(output_dir=str(tmp_path / "out")), SimpleNamespace(global_step=7), None)
    assert (ck / ".complete").exists()
