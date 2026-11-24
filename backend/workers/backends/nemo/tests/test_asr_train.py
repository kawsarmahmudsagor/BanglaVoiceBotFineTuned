import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.contract import read_config
from mlops_backend import asr

INDIC = "ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large"
NEMOTRON = "nvidia/nemotron-3.5-asr-streaming-0.6b"


def test_nemo_init_path_choice(tmp_path):
    d = tmp_path / "init"
    assert asr.init_path_choice({"format": "nemo"}, d) == ("restore_from", d / "model.nemo")
    assert asr.init_path_choice({"format": "lightning-ckpt"}, d) == ("state_dict", d / "model.ckpt")
    with pytest.raises(ValueError, match="hf-full"):
        asr.init_path_choice({"format": "hf-full"}, d)
    with pytest.raises(ValueError, match="init_meta"):
        asr.init_path_choice(None, d)


def test_data_config_paths(tmp_path):
    cfg = asr.data_config(tmp_path / "train.jsonl", batch_size=4, shuffle=True, cpu_count=32)
    assert cfg == {"manifest_filepath": str(tmp_path / "train.jsonl"), "sample_rate": 16000, "batch_size": 4,
                   "shuffle": True, "num_workers": 8, "max_duration": 30, "pin_memory": True}
    assert asr.data_config(tmp_path / "v.jsonl", batch_size=1, shuffle=False, cpu_count=2)["num_workers"] == 2


class FakeModel:
    def __init__(self):
        self.cfg = {"aux_ctc": {"ctc_loss_weight": 0.3},
                    "encoder": {"att_context_size": [70, 13], "conv_context_size": "causal"}}
        self.calls = []

    def setup_training_data(self, c):
        self.calls.append(("train", c))

    def setup_validation_data(self, c):
        self.calls.append(("val", c))


def test_streaming_keys_untouched(tmp_path):
    (tmp_path / "train.jsonl").write_text("{}\n")
    (tmp_path / "val.jsonl").write_text("{}\n")
    m = FakeModel()
    before = json.dumps(m.cfg, sort_keys=True)
    has_val = asr.apply_data_config(m, tmp_path, batch_size=2)
    assert has_val is True
    assert json.dumps(m.cfg, sort_keys=True) == before  # aux_ctc weight and encoder streaming keys kept
    assert [k for k, _ in m.calls] == ["train", "val"]
    assert m.calls[0][1]["manifest_filepath"] == str(tmp_path / "train.jsonl")
    assert m.calls[0][1]["shuffle"] is True and m.calls[1][1]["shuffle"] is False


def test_no_val_manifest(tmp_path):
    (tmp_path / "train.jsonl").write_text("{}\n")
    (tmp_path / "val.jsonl").write_text("")
    m = FakeModel()
    assert asr.apply_data_config(m, tmp_path, batch_size=2) is False
    assert [k for k, _ in m.calls] == ["train"]


def test_merged_optim_keeps_scheduler():
    base = {"name": "adamw", "lr": 1e-3, "betas": [0.9, 0.98], "sched": {"name": "CosineAnnealing", "warmup_ratio": 0.1,
                                                                      "min_lr": 1e-6}}
    out = asr.merged_optim(base, {"learning_rate": 5e-5, "warmup_steps": 7}, 1000)
    assert out["lr"] == 5e-5 and out["betas"] == [0.9, 0.98]
    assert out["sched"]["name"] == "CosineAnnealing" and out["sched"]["warmup_steps"] == 7
    assert "warmup_ratio" not in out["sched"]
    assert base["lr"] == 1e-3 and "warmup_ratio" in base["sched"]  # input untouched
    assert asr.merged_optim(base, {}, 1000) == base  # ratio 0.1 * 1000 = 100 = cap: unchanged


def test_merged_optim_clamps_warmup_without_recipe_value():
    sched = {"name": "CosineAnnealing", "warmup_steps": 10000, "min_lr": 1e-6}
    out = asr.merged_optim({"lr": 1e-3, "sched": sched}, {}, 20)
    assert out["sched"]["warmup_steps"] == 2
    ratio = asr.merged_optim({"sched": {"warmup_ratio": 0.5}}, {}, 20)
    assert ratio["sched"] == {"warmup_steps": 2}
    small = asr.merged_optim({"sched": {"warmup_steps": 1}}, {}, 20)
    assert small["sched"]["warmup_steps"] == 1


def test_merged_optim_without_sched():
    assert asr.merged_optim({"lr": 1.0}, {"learning_rate": 2e-5, "warmup_steps": 3}, 10) == {"lr": 2e-5}


def test_merged_optim_on_real_dictconfig():
    omegaconf = pytest.importorskip("omegaconf")
    cfg = omegaconf.OmegaConf.create({"optim": {"lr": 1e-3, "sched": {"warmup_ratio": 0.5}}})
    out = asr.merged_optim(omegaconf.OmegaConf.to_container(cfg.optim, resolve=True), {"learning_rate": 1e-5}, 20)
    assert out["lr"] == 1e-5 and out["sched"]["warmup_steps"] == 2


def test_save_interval():
    assert asr.save_interval({"save_steps": 5}, 10) == 5
    assert asr.save_interval({}, 100) == 50
    assert asr.save_interval({}, 10000) == 1000


def test_hyp_texts():
    hyp = SimpleNamespace(text="b")
    assert asr.hyp_texts(["a", hyp]) == ["a", "b"]
    assert asr.hyp_texts((["a", "b"], [["a2"], ["b2"]])) == ["a", "b"]  # RNNT (best, all) tuple
    assert asr.hyp_texts([[hyp, SimpleNamespace(text="c")]]) == ["b"]  # n-best lists


def test_exporter_copies_final(tmp_path, run_dir):
    rd = run_dir("asr", NEMOTRON, export=["nemo"])
    cfg = read_config(rd)
    final = rd.checkpoints_dir / "final"
    final.mkdir(parents=True)
    (final / "model.nemo").write_bytes(b"nemo")
    ex = asr.nemo_exporters(rd)
    with pytest.raises(RuntimeError, match="no completed final"):
        out = rd.export_dir / "nemo"
        out.mkdir(parents=True)
        ex["nemo"](out)
    (final / ".complete").write_text("")
    arts = ex["nemo"](out)
    assert (out / "model.nemo").read_bytes() == b"nemo"
    assert arts[0].format == "nemo" and arts[0].serve_with == "nemo" and arts[0].path == "nemo"


def test_handlers_exposed():
    assert callable(asr.train) and callable(asr.eval) and callable(asr.export)


@pytest.mark.heavy
@pytest.mark.parametrize("model", [INDIC, NEMOTRON])
def test_nemo_20_steps(run_dir, monkeypatch, tmp_path, model):
    """GPU VM only: 20 steps on the tiny dataset, eval with baseline, .nemo export."""
    from app.core.backend_kit import Reporter
    from app.core.contract import read_eval_results

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    rd = run_dir("asr", model, {"max_steps": 20, "batch_size": 1, "grad_accum": 1, "learning_rate": 1e-5},
                 method={"type": "full"}, export=["nemo"])
    cfg = read_config(rd)
    rep = Reporter(rd)
    assert asr.prepare(rd, cfg, rep) == 0
    assert asr.train(rd, cfg, rep) == 0
    final = rd.checkpoints_dir / "final"
    assert (final / ".complete").exists() and (final / "model.nemo").exists()
    assert json.loads((final / "init_meta.json").read_text())["format"] == "nemo"
    assert asr.eval(rd, cfg, rep) == 0
    res = read_eval_results(rd).metrics
    assert "wer" in res and "base/wer" in res
    assert asr.export(rd, cfg, rep) == 0
    assert (rd.export_dir / "nemo" / "model.nemo").exists()
    assert rd.export_manifest_path.exists()


@pytest.mark.heavy
def test_nemo_checkpoints_resume_and_warm_start(run_dir, monkeypatch, tmp_path):
    """GPU VM only: real Lightning checkpoints (save_steps=5), resume to step 15, warm start from lightning-ckpt."""
    from app.core.backend_kit import Reporter, latest_checkpoint, read_init_meta
    from app.core.contract import InitRef, RunDir, write_config

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    hp = {"max_steps": 10, "save_steps": 5, "batch_size": 1, "grad_accum": 1, "learning_rate": 1e-5}
    rd = run_dir("asr", NEMOTRON, hp, method={"type": "full"})
    cfg = read_config(rd)
    rep = Reporter(rd)
    assert asr.prepare(rd, cfg, rep) == 0
    assert asr.train(rd, cfg, rep) == 0
    for n in (5, 10):
        d = rd.checkpoints_dir / f"step-{n}"
        assert (d / "model.ckpt").is_file() and (d / ".complete").exists()
        assert read_init_meta(d)["format"] == "lightning-ckpt"
    assert latest_checkpoint(rd).name == "step-10"

    cfg2 = cfg.model_copy(update={"resume": True, "hparams": {**hp, "max_steps": 15}})
    assert asr.train(rd, cfg2, rep) == 0  # continues from step-10, not from scratch
    assert read_init_meta(rd.checkpoints_dir / "final")["step"] == 15

    # warm start: a second run initialised (read-only, hard-linked in real runs) from step-10's lightning ckpt
    rd2 = RunDir(tmp_path / "run2")
    rd2.ensure()
    rd2.init_dir.mkdir(parents=True, exist_ok=True)
    src = rd.checkpoints_dir / "step-10"
    for f in ("model.ckpt", "init_meta.json"):
        shutil.copy2(src / f, rd2.init_dir / f)
    cfg3 = cfg.model_copy(update={"run_id": "r2", "resume": False,
                                  "init_from": InitRef(path=str(rd2.init_dir), source_run="r1", checkpoint="step-10", method="full", format="lightning-ckpt"),
                                  "hparams": {**hp, "max_steps": 5}})
    write_config(rd2, cfg3)
    shutil.copy2(rd.prepared_refs_path, rd2.prepared_refs_path)
    assert asr.train(rd2, cfg3, rep) == 0
    assert (rd2.checkpoints_dir / "final" / "model.nemo").exists()
