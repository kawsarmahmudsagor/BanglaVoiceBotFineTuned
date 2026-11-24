import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from app.core.contract import RunConfig, RunDir, write_config
from mlops_backend.models import MODELS

CAPS = json.loads((Path(__file__).parents[1] / "capabilities.json").read_text(encoding="utf-8"))
MMS = "facebook/mms-1b-all"


@pytest.mark.parametrize("task", ["asr", "tts"])
def test_capabilities_schema_valid(task):
    Draft202012Validator.check_schema(CAPS["tasks"][task]["schema"])


@pytest.mark.parametrize("task", ["asr", "tts"])
def test_base_model_enum_matches_registry(task):
    enum = CAPS["tasks"][task]["schema"]["properties"]["base_model"]["enum"]
    assert sorted(enum) == sorted(k for k, v in MODELS.items() if v.task == task)


def test_methods_and_exports():
    asr, tts = CAPS["tasks"]["asr"], CAPS["tasks"]["tts"]
    assert asr["methods"] == ["full", "adapter"]
    assert asr["export_formats"] == ["safetensors", "ct2", "adapter-safetensors"]
    assert tts["methods"] == ["full"]
    assert tts["export_formats"] == ["safetensors"]


def test_adapter_only_for_mms():
    v = Draft202012Validator(CAPS["tasks"]["asr"]["schema"])
    adapter = {"type": "adapter"}
    assert not list(v.iter_errors({"base_model": MMS, "method": adapter, "hparams": {}}))
    assert list(v.iter_errors({"base_model": "openai/whisper-large-v3-turbo", "method": adapter, "hparams": {}}))
    assert not list(v.iter_errors(
        {"base_model": "openai/whisper-large-v3-turbo", "method": {"type": "full"}, "hparams": {}}))


def test_parler_description_template():
    schema = CAPS["tasks"]["tts"]["schema"]
    prop = schema["properties"]["hparams"]["properties"]["description_template"]
    assert prop["type"] == "string"
    assert prop["default"] == "{speaker} speaks clearly in a studio-quality recording."
    v = Draft202012Validator(schema)
    assert not list(v.iter_errors({"method": {"type": "full"}, "hparams": {"description_template": "x"}}))
    assert list(v.iter_errors({"method": {"type": "full"}, "hparams": {"description_template": 3}}))


def test_phase_dispatch_unknown_model_fails(tmp_path):
    rd = RunDir(tmp_path / "run")
    rd.ensure()
    write_config(rd, RunConfig(
        run_id="r1", attempt=1, resume=False, task="asr", backend="hf", base_model="x/y",
        method={"type": "full"}, hparams={}, export=[], seed=1, datasets={}, prepared_root="prepared"))
    r = subprocess.run([sys.executable, "-m", "mlops_backend", "train", "--run-dir", str(rd.path)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 1
    assert "unsupported base_model 'x/y' for backend hf" in r.stderr


def test_phase_not_implemented_is_clean(tmp_path):
    rd = RunDir(tmp_path / "run")
    rd.ensure()
    write_config(rd, RunConfig(
        run_id="r1", attempt=1, resume=False, task="asr", backend="hf",
        base_model="openai/whisper-large-v3-turbo",
        method={"type": "full"}, hparams={}, export=[], seed=1, datasets={}, prepared_root="prepared"))
    r = subprocess.run([sys.executable, "-m", "mlops_backend", "train", "--run-dir", str(rd.path)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 1
    assert "Traceback" not in r.stderr
    assert "not installed" in r.stderr or "not implemented" in r.stderr
