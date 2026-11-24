import json
import subprocess
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

from app.core.contract import RunConfig, RunDir, write_config
from mlops_backend.models import MODELS

CAPS = json.loads((Path(__file__).parents[1] / "capabilities.json").read_text(encoding="utf-8"))
TTS = CAPS["tasks"]["tts"]


def test_capabilities_schema_valid():
    Draft202012Validator.check_schema(TTS["schema"])


def test_base_model_enum_matches_registry():
    enum = TTS["schema"]["properties"]["base_model"]["enum"]
    assert sorted(enum) == sorted(k for k, v in MODELS.items() if v.task == "tts")
    assert MODELS["ai4bharat/IndicF5"].gated is True
    assert MODELS["ai4bharat/IndicF5"].family == "f5"


def test_methods_exports_and_hparams():
    assert list(CAPS["tasks"]) == ["tts"]
    assert TTS["methods"] == ["full"]
    assert TTS["export_formats"] == ["f5-ckpt"]
    hp = TTS["schema"]["properties"]["hparams"]["properties"]
    assert set(hp) == {"learning_rate", "epochs", "batch_size_frames", "grad_accum", "max_steps",
                       "warmup_steps", "save_steps", "seed", "eval", "prepare"}
    assert hp["batch_size_frames"]["default"] == 3200
    assert hp["warmup_steps"]["default"] == 1000


def _cfg(model):
    return RunConfig(
        run_id="r1", attempt=1, resume=False, task="tts", backend="f5", base_model=model,
        method={"type": "full"}, hparams={}, export=[], seed=1, datasets={}, prepared_root="prepared")


def test_phase_dispatch_unknown_model_fails(tmp_path):
    rd = RunDir(tmp_path / "run")
    rd.ensure()
    write_config(rd, _cfg("x/y"))
    r = subprocess.run([sys.executable, "-m", "mlops_backend", "train", "--run-dir", str(rd.path)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 1
    assert "unsupported base_model 'x/y' for backend f5" in r.stderr


def test_phase_not_implemented_is_clean(tmp_path, monkeypatch, capsys):
    from app.core.contract import Phase
    from mlops_backend import phases

    monkeypatch.setitem(phases._FAMILY_MODULE, "f5", "_no_such_module")
    rd = RunDir(tmp_path / "run")
    rd.ensure()
    cfg = _cfg("ai4bharat/IndicF5")
    assert phases._dispatch(Phase.train)(rd, cfg, None) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "support not installed" in err
