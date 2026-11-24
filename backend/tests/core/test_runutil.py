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


def test_val_subset_is_seeded_and_capped():
    from app.core.backend_kit.runutil import MAX_VAL_SAMPLES, max_val_samples, val_subset_indices
    assert MAX_VAL_SAMPLES == 2000 and max_val_samples({}) == 2000
    assert max_val_samples({"eval": {"max_val_samples": 50}}) == 50
    assert val_subset_indices(10, 20, seed=1) is None  # small val: all rows
    a = val_subset_indices(1000, 100, seed=7)
    assert a == val_subset_indices(1000, 100, seed=7)  # deterministic for a seed
    assert a != val_subset_indices(1000, 100, seed=8)
    assert len(a) == 100 == len(set(a)) and a == sorted(a) and all(0 <= i < 1000 for i in a)


def test_cap_val_selects_subset():
    from app.core.backend_kit.runutil import cap_val

    class DS:
        def __init__(self, n):
            self.rows = list(range(n))

        def __len__(self):
            return len(self.rows)

        def select(self, idx):
            d = DS(0)
            d.rows = [self.rows[i] for i in idx]
            return d

    cfg = _cfg(eval={"max_val_samples": 4})
    small = DS(3)
    assert cap_val(small, cfg) is small
    out = cap_val(DS(50), cfg)
    assert len(out) == 4 and out.rows == cap_val(DS(50), cfg).rows
