import json
import subprocess
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

from app.core.contract import RunConfig, RunDir, write_config
from mlops_backend.models import MODELS

CAPS = json.loads((Path(__file__).parents[1] / "capabilities.json").read_text(encoding="utf-8"))
ASR = CAPS["tasks"]["asr"]


def test_capabilities_schema_valid():
    Draft202012Validator.check_schema(ASR["schema"])


def test_base_model_enum_matches_registry():
    enum = ASR["schema"]["properties"]["base_model"]["enum"]
    assert sorted(enum) == sorted(k for k, v in MODELS.items() if v.task == "asr")
    assert MODELS["ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large"].gated is True


def test_methods_exports_and_hparams():
    assert list(CAPS["tasks"]) == ["asr"]
    assert ASR["methods"] == ["full"]
    assert ASR["export_formats"] == ["nemo"]
    hp = ASR["schema"]["properties"]["hparams"]["properties"]
    assert set(hp) == {"learning_rate", "epochs", "batch_size", "grad_accum", "max_steps", "warmup_steps", "save_steps", "keep_checkpoints", "seed", "eval", "prepare"}


def _cfg(model):
    return RunConfig(
        run_id="r1", attempt=1, resume=False, task="asr", backend="nemo", base_model=model,
        method={"type": "full"}, hparams={}, export=[], seed=1, datasets={}, prepared_root="prepared")


def test_phase_dispatch_unknown_model_fails(tmp_path):
    rd = RunDir(tmp_path / "run")
    rd.ensure()
    write_config(rd, _cfg("x/y"))
    r = subprocess.run([sys.executable, "-m", "mlops_backend", "train", "--run-dir", str(rd.path)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 1
    assert "unsupported base_model 'x/y' for backend nemo" in r.stderr


def test_phase_not_implemented_is_clean(tmp_path, monkeypatch, capsys):
    from app.core.contract import Phase
    from mlops_backend import phases

    monkeypatch.setitem(phases._FAMILY_MODULE, "nemo-manifest", "_no_such_module")
    rd = RunDir(tmp_path / "run")
    rd.ensure()
    cfg = _cfg("bengaliAI/BanglaConformer")
    assert phases._dispatch(Phase.train)(rd, cfg, None) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "support not installed" in err


def test_every_task_has_keep_checkpoints():
    for task in CAPS["tasks"].values():
        kc = task["schema"]["properties"]["hparams"]["properties"]["keep_checkpoints"]
        assert kc == {"type": "integer", "minimum": 1, "default": 2}


def test_eval_documents_max_val_samples():
    for task in CAPS["tasks"].values():
        ev = task["schema"]["properties"]["hparams"]["properties"]["eval"]
        assert ev["type"] == "object"
        mvs = ev["properties"]["max_val_samples"]
        assert (mvs["type"], mvs["minimum"], mvs["default"]) == ("integer", 1, 2000)
        assert {"baseline", "max_samples"} <= set(ev["properties"])
