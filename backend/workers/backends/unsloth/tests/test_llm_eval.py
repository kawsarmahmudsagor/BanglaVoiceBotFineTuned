import json

import pytest

from app.core.contract import read_config
from mlops_backend.llm import _baseline_eval_key, eval_llm, eval_loss_files, eval_rows

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


def test_eval_loss_files(run_dir):
    rd = run_dir("llm-sft", QWEN)
    cfg = read_config(rd)
    rd.prepared_refs_path.write_text(json.dumps([
        {"role": "train", "path": "/p/train"}, {"role": "eval", "path": "/p/eval"}]))
    files = eval_loss_files(rd, cfg)
    assert [f.parts[-2:] for f in files] == [("eval", "train.jsonl"), ("eval", "val.jsonl"), ("eval", "test.jsonl")]
    cfg2 = cfg.model_copy(update={"datasets": {"train": cfg.datasets["train"]}})
    assert [f.name for f in eval_loss_files(rd, cfg2)] == ["val.jsonl"]
    assert files[0].parent.name == "eval"


def test_baseline_eval_key_varies(run_dir):
    rd = run_dir("llm-sft", QWEN)
    cfg = read_config(rd)
    a = _baseline_eval_key(cfg, 200, 2048)
    assert a == {"max_samples": 200, "max_seq_len": 2048, "load_in_4bit": False}
    assert _baseline_eval_key(cfg, 3, 2048) != a
    assert _baseline_eval_key(cfg, 200, 256) != a
    q = cfg.model_copy(update={"method": {"type": "qlora"}})
    assert _baseline_eval_key(q, 200, 2048)["load_in_4bit"] is True
    from app.core.contract import InitRef
    w = cfg.model_copy(update={"init_from": InitRef(path="p", source_run="r0", checkpoint="final",
                                                    method="lora", format="hf-peft")})
    assert _baseline_eval_key(w, 200, 2048)["init"] == ["r0", "final"]


def test_dispatch_handlers_resolve():
    from app.core.contract import Phase
    from mlops_backend import llm
    from mlops_backend.phases import HANDLERS
    assert callable(llm.prepare) and callable(llm.export) and llm.train is llm.train_llm and llm.eval is llm.eval_llm
    assert all(Phase(p) in HANDLERS for p in ("prepare", "train", "eval", "export"))


def test_eval_requires_complete_final(run_dir):
    rd = run_dir("llm-sft", QWEN)
    (rd.checkpoints_dir / "final").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no completed final checkpoint"):
        eval_llm(rd, read_config(rd), None)


def _heavy_run(run_dir, hparams):
    from app.core.backend_kit import Reporter
    from app.core.contract import read_config, write_config
    from mlops_backend.llm import train_llm
    rd = run_dir("llm-sft", "Qwen/Qwen3-4B-Instruct-2507",
                 {"max_steps": 10, "max_seq_len": 256, "batch_size": 1, "grad_accum": 1, **hparams})
    cfg = read_config(rd)
    cfg = cfg.model_copy(update={"datasets": {"train": cfg.datasets["train"]}})  # eval falls back to train `val`
    write_config(rd, cfg)
    prep = rd.path / "prep"
    prep.mkdir()
    row = json.dumps({"input_ids": list(range(1, 33)), "labels": [-100] * 16 + list(range(17, 33))})
    for sp in ("train", "val"):
        (prep / f"{sp}.jsonl").write_text(chr(10).join([row] * 8))
    rd.prepared_refs_path.write_text(json.dumps([
        {"role": "train", "path": str(prep), "counts": {"train": 8, "val": 8}}]))
    assert train_llm(rd, cfg, Reporter(rd)) == 0
    return rd, cfg, Reporter(rd)


@pytest.mark.heavy
def test_llm_eval_writes_results_and_samples(run_dir):
    from app.core.contract import read_eval_results
    rd, cfg, rep = _heavy_run(run_dir, {"eval": {"max_samples": 3}})
    assert eval_llm(rd, cfg, rep) == 0
    m = read_eval_results(rd).metrics
    for k in ("eval_loss", "perplexity", "bangla_script_share", "base/eval_loss", "base/bangla_script_share"):
        assert k in m
    lines = rd.eval_samples_path.read_text(encoding="utf-8").splitlines()
    assert 1 <= len(lines) <= 3


@pytest.mark.heavy
def test_llm_eval_baseline_disabled(run_dir):
    from app.core.contract import read_eval_results
    rd, cfg, rep = _heavy_run(run_dir, {"eval": {"max_samples": 3, "baseline": False}})
    assert eval_llm(rd, cfg, rep) == 0
    m = read_eval_results(rd).metrics
    assert "eval_loss" in m and not any(k.startswith("base/") for k in m)
