import json
from types import SimpleNamespace

from app.core.backend_kit import StopFlag
from app.core.backend_kit.runutil import stopped_early, trainer_args
from app.core.contract import RunConfig, RunDir


def _cfg(**hp):
    return RunConfig(run_id="r", attempt=1, resume=False, task="asr", backend="hf", base_model="m",
                     method={}, hparams=hp, export=[], seed=1, datasets={}, prepared_root="/x")


def _rd(tmp_path):
    rd = RunDir(tmp_path)
    rd.ensure()
    rd.prepared_refs_path.write_text(json.dumps([{"role": "train", "path": "p", "counts": {"train": 100}}]))
    return rd


def test_trainer_args_save_total_limit_from_keep_checkpoints(tmp_path):
    rd = _rd(tmp_path)
    assert trainer_args(_cfg(), rd, "adamw_torch")["save_total_limit"] == 2
    assert trainer_args(_cfg(keep_checkpoints=4), rd, "adamw_torch")["save_total_limit"] == 4
    assert trainer_args(_cfg(keep_checkpoints=0), rd, "adamw_torch")["save_total_limit"] == 1


def _trainer(step, max_steps):
    return SimpleNamespace(state=SimpleNamespace(global_step=step, max_steps=max_steps))


def test_stopped_early_only_before_the_natural_end():
    stop = StopFlag()
    assert not stopped_early(stop, _trainer(5, 10))
    stop.requested = True
    assert stopped_early(stop, _trainer(5, 10))
    assert not stopped_early(stop, _trainer(10, 10))  # SIGTERM after training finished: write final, exit 0
    assert stopped_early(stop, _trainer(0, 0))  # unknown max_steps: treat as early
