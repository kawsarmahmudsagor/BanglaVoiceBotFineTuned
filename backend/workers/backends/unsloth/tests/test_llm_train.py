import json

import pytest

from app.core.backend_kit import read_init_meta
from mlops_backend.llm import _trainer_args, plan_init, total_steps, write_ckpt_meta

QWEN = "Qwen/Qwen3-4B-Instruct-2507"


def _refs(rd, n=16):
    rd.prepared_refs_path.write_text(json.dumps([
        {"role": "train", "path": str(rd.path / "prep"), "counts": {"train": n, "val": 2}},
    ]))


def test_trainer_args_from_hparams(run_dir):
    rd = run_dir("llm-sft", QWEN, {"learning_rate": 1e-4, "epochs": 3, "batch_size": 2, "grad_accum": 2,
                                   "warmup_steps": 5}, method={"type": "full"})
    _refs(rd, 16)
    cfg = __import__("app.core.contract", fromlist=["read_config"]).read_config(rd)
    a = _trainer_args(cfg, rd)
    assert a["output_dir"] == str(rd.checkpoints_dir)
    assert a["optim"] == "adamw_8bit"
    assert a["num_train_epochs"] == 3
    assert "max_steps" not in a or a["max_steps"] == -1
    assert a["bf16"] is True and a["report_to"] == [] and a["save_total_limit"] == 3
    assert a["save_steps"] == 50 and a["eval_steps"] == 50 and a["eval_strategy"] == "steps"
    assert a["per_device_train_batch_size"] == 2 and a["gradient_accumulation_steps"] == 2


def test_trainer_args_max_steps_overrides_epochs(run_dir):
    from app.core.contract import read_config
    rd = run_dir("llm-sft", QWEN, {"max_steps": 20, "epochs": 3})
    _refs(rd)
    a = _trainer_args(read_config(rd), rd)
    assert a["max_steps"] == 20
    assert "num_train_epochs" not in a
    assert a["optim"] == "adamw_torch"


def test_total_steps_and_save_steps(run_dir):
    from app.core.contract import read_config
    rd = run_dir("llm-sft", QWEN, {"epochs": 10, "batch_size": 1, "grad_accum": 1})
    _refs(rd, 1000)
    cfg = read_config(rd)
    assert total_steps(cfg, rd) == 10000
    assert _trainer_args(cfg, rd)["save_steps"] == 1000


@pytest.mark.parametrize("meta,method,expected", [
    (None, "lora", "base"),
    (None, "full", "base"),
    ({"format": "hf-peft"}, "lora", "attach_adapter"),
    ({"format": "hf-peft"}, "qlora", "attach_adapter"),
    ({"format": "hf-peft"}, "full", "merge_adapter"),
    ({"format": "hf-full"}, "full", "full_weights"),
    ({"format": "hf-full"}, "lora", "full_then_adapter"),
    ({"format": "hf-full"}, "qlora", "full_then_adapter"),
])
def test_plan_init(meta, method, expected):
    assert plan_init(meta, method) == expected


def test_plan_init_rejects_foreign_format():
    with pytest.raises(ValueError):
        plan_init({"format": "nemo"}, "lora")


@pytest.mark.parametrize("method,fmt", [("lora", "hf-peft"), ("qlora", "hf-peft"), ("full", "hf-full")])
def test_write_ckpt_meta(run_dir, method, fmt):
    from app.core.contract import read_config
    rd = run_dir("llm-sft", QWEN, method={"type": method})
    d = rd.checkpoints_dir / "checkpoint-7"
    d.mkdir(parents=True)
    write_ckpt_meta(d, read_config(rd), 7)
    m = read_init_meta(d)
    assert m["format"] == fmt and m["backend"] == "unsloth" and m["step"] == 7
    assert m["base_model"] == QWEN and m["method"] == method


@pytest.mark.heavy
def test_llm_train_20_steps_tiny(run_dir):
    from app.core.backend_kit import Reporter
    from app.core.contract import read_config
    from mlops_backend.llm import train_llm
    rd = run_dir("llm-sft", QWEN, {"max_steps": 20, "max_seq_len": 256, "batch_size": 1, "grad_accum": 1,
                                   "lora_r": 8, "lora_alpha": 16, "learning_rate": 2e-4})
    prep = rd.path / "prep"
    prep.mkdir()
    row = json.dumps({"input_ids": list(range(1, 33)), "labels": [-100] * 16 + list(range(17, 33))})
    for sp in ("train", "val"):
        (prep / f"{sp}.jsonl").write_text("\n".join([row] * 16))
    _refs(rd, 16)
    assert train_llm(rd, read_config(rd), Reporter(rd)) == 0
    assert (rd.checkpoints_dir / "final" / ".complete").exists()
    assert any(json.loads(l).get("name") == "loss" for l in rd.metrics_path.read_text().splitlines())
