import json
import subprocess
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

from app.core.contract import RunConfig, RunDir, write_config
from mlops_backend.models import MODELS

ROOT = Path(__file__).parents[1]
CAPS = json.loads((ROOT / "capabilities.json").read_text(encoding="utf-8"))
TTS = CAPS["tasks"]["tts"]
REPO = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"


def test_capabilities_schema_valid():
    Draft202012Validator.check_schema(TTS["schema"])


def test_base_model_enum_matches_registry():
    enum = TTS["schema"]["properties"]["base_model"]["enum"]
    assert sorted(enum) == sorted(k for k, v in MODELS.items() if v.task == "tts")
    assert enum == [REPO]
    assert MODELS[REPO].gated is False
    assert MODELS[REPO].family == "cosyvoice"


def test_methods_exports_and_hparams():
    assert list(CAPS["tasks"]) == ["tts"]
    assert TTS["methods"] == ["full"]
    assert TTS["export_formats"] == ["cosyvoice-dir"]
    hp = TTS["schema"]["properties"]["hparams"]["properties"]
    assert set(hp) == {"components", "learning_rate", "epochs", "max_frames_in_batch", "grad_accum",
                       "warmup_steps", "max_steps", "save_steps", "keep_checkpoints", "seed", "eval",
                       "prepare"}
    comp = hp["components"]
    assert comp["type"] == "array" and comp["minItems"] == 1 and comp["uniqueItems"] is True
    assert comp["items"]["enum"] == ["llm", "flow", "hifigan"]
    assert comp["default"] == ["llm", "flow"]
    assert hp["learning_rate"]["default"] == 1e-5
    assert hp["epochs"]["default"] == 10
    assert hp["max_frames_in_batch"]["default"] == 2000
    assert hp["grad_accum"]["default"] == 2
    assert hp["warmup_steps"]["default"] == 2500


def test_upstream_commit_pinned():
    lines = (ROOT / "UPSTREAM_COMMIT").read_text(encoding="utf-8").splitlines()
    assert len(lines[0]) == 40 and all(c in "0123456789abcdef" for c in lines[0])


def _cfg(model):
    return RunConfig(
        run_id="r1", attempt=1, resume=False, task="tts", backend="cosyvoice", base_model=model,
        method={"type": "full"}, hparams={}, export=[], seed=1, datasets={}, prepared_root="prepared")


def test_phase_dispatch_unknown_model_fails(tmp_path):
    rd = RunDir(tmp_path / "run")
    rd.ensure()
    write_config(rd, _cfg("x/y"))
    r = subprocess.run([sys.executable, "-m", "mlops_backend", "train", "--run-dir", str(rd.path)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 1
    assert "unsupported base_model 'x/y' for backend cosyvoice" in r.stderr


def test_phase_not_implemented_is_clean(tmp_path, monkeypatch, capsys):
    from app.core.contract import Phase
    from mlops_backend import phases

    monkeypatch.setitem(phases._FAMILY_MODULE, "cosyvoice", "_no_such_module")
    rd = RunDir(tmp_path / "run")
    rd.ensure()
    cfg = _cfg(REPO)
    assert phases._dispatch(Phase.train)(rd, cfg, None) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "support not installed" in err
