import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from app.core.contract import RunConfig, RunDir, write_config
from mlops_backend.models import MODELS

CAPS = json.loads((Path(__file__).parents[1] / "capabilities.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("task", ["llm-sft", "asr"])
def test_capabilities_schema_valid(task):
    Draft202012Validator.check_schema(CAPS["tasks"][task]["schema"])


@pytest.mark.parametrize("task", ["llm-sft", "asr"])
def test_base_model_enum_matches_registry(task):
    enum = CAPS["tasks"][task]["schema"]["properties"]["base_model"]["enum"]
    assert sorted(enum) == sorted(k for k, v in MODELS.items() if v.task == task)


def test_full_requires_gradient_checkpointing():
    v = Draft202012Validator(CAPS["tasks"]["llm-sft"]["schema"])
    bad = {"method": {"type": "full"}, "hparams": {"gradient_checkpointing": False}}
    ok = {"method": {"type": "full"}, "hparams": {"gradient_checkpointing": True}}
    assert list(v.iter_errors(bad)) and not list(v.iter_errors(ok))
    assert not list(v.iter_errors({"method": {"type": "lora"}, "hparams": {"lora_r": 16}}))


def test_phase_dispatch_unknown_model_fails(tmp_path):
    rd = RunDir(tmp_path / "run")
    rd.ensure()
    write_config(rd, RunConfig(
        run_id="r1", attempt=1, resume=False, task="llm-sft", backend="unsloth", base_model="x/y",
        method={"type": "lora"}, hparams={}, export=[], seed=1, datasets={}, prepared_root="prepared"))
    r = subprocess.run([sys.executable, "-m", "mlops_backend", "train", "--run-dir", str(rd.path)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 1
    assert "unsupported base_model 'x/y' for backend unsloth" in r.stderr


def test_every_task_has_keep_checkpoints():
    for task in CAPS["tasks"].values():
        kc = task["schema"]["properties"]["hparams"]["properties"]["keep_checkpoints"]
        assert kc == {"type": "integer", "minimum": 1, "default": 2}
