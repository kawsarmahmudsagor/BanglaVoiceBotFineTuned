import json

import pytest

from app.core.contract import read_config
from mlops_backend.llm import eval_loss_files, eval_rows

QWEN = "Qwen/Qwen3-4B-Instruct-2507"


def test_eval_rows_fallback_to_val(run_dir):
    rd = run_dir("llm-sft", QWEN)
    cfg = read_config(rd)
    cfg = cfg.model_copy(update={"datasets": {"train": cfg.datasets["train"]}})
    rows = eval_rows(cfg)
    assert [r["sample_id"] for r in rows] == ["s2"]  # the only `val` row of the fixture


def test_eval_rows_use_eval_dataset_splits(run_dir):
    rd = run_dir("llm-sft", QWEN)
    cfg = read_config(rd)
    ev = cfg.datasets["eval"].model_copy(update={"splits": ["test"], "split": None})
    cfg = cfg.model_copy(update={"datasets": {**cfg.datasets, "eval": ev}})
    assert len(eval_rows(cfg)) == 4  # eval-only dataset: every row for the requested split


def test_eval_loss_files(run_dir, tmp_path):
    rd = run_dir("llm-sft", QWEN)
    cfg = read_config(rd)
    rd.prepared_refs_path.write_text(json.dumps([
        {"role": "train", "path": "/p/train"}, {"role": "eval", "path": "/p/eval"}]))
    files = eval_loss_files(rd, cfg)
    assert [f.parts[-2:] for f in files] == [("eval", "train.jsonl"), ("eval", "val.jsonl"), ("eval", "test.jsonl")]
    cfg2 = cfg.model_copy(update={"datasets": {"train": cfg.datasets["train"]}})
    assert [f.name for f in eval_loss_files(rd, cfg2)] == ["val.jsonl"]
    assert files[0].parent.name == "eval"


@pytest.mark.heavy
def test_llm_eval_writes_results_and_samples(run_dir):
    pytest.skip("runs on the GPU VM via scripts/test_backend.sh unsloth --heavy")
