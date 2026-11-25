import json
import subprocess
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

from app.core.contract import RunConfig, RunDir, write_config
from mlops_backend.models import MODELS

ROOT = Path(__file__).parents[1]
REPO = Path(__file__).resolve().parents[5]
CAPS = json.loads((ROOT / "capabilities.json").read_text(encoding="utf-8"))
ASR = CAPS["tasks"]["asr"]
MODEL = "SayedShaun/Qwen3-ASR-0.6B-Bengali"


def test_capabilities_schema_valid():
    assert CAPS["name"] == "qwen-asr" == ROOT.name
    Draft202012Validator.check_schema(ASR["schema"])


def test_base_model_enum_matches_registry():
    enum = ASR["schema"]["properties"]["base_model"]["enum"]
    assert sorted(enum) == sorted(k for k, v in MODELS.items() if v.task == "asr")
    assert enum[0] == MODEL
    assert all(v.family == "qwen-asr" and not v.gated for v in MODELS.values())


def test_methods_exports_and_hparams():
    assert list(CAPS["tasks"]) == ["asr"]
    assert ASR["methods"] == ["full"]
    assert ASR["export_formats"] == ["safetensors"]
    assert ASR["serve_formats"] == []
    hp = ASR["schema"]["properties"]["hparams"]["properties"]
    assert set(hp) == {"learning_rate", "epochs", "max_steps", "batch_size", "grad_accum", "warmup_ratio",
                       "save_steps", "max_label_tokens", "gradient_checkpointing", "freeze_audio_encoder",
                       "keep_checkpoints", "seed", "eval", "prepare"}
    lr = hp["learning_rate"]
    assert (lr["default"], lr["minimum"], lr["maximum"]) == (2e-5, 1e-6, 1e-3)
    assert hp["batch_size"]["default"] == 8 and hp["batch_size"]["maximum"] == 16
    assert hp["warmup_ratio"]["default"] == 0.02
    assert hp["max_label_tokens"]["default"] == 448
    assert hp["gradient_checkpointing"]["default"] is True
    assert hp["freeze_audio_encoder"]["default"] is False


def test_batch_size_fits_a_32gb_card():
    """Logits over the 151,936-token vocabulary dominate memory: batch 32 cannot fit; grad_accum scales instead."""
    bs = ASR["schema"]["properties"]["hparams"]["properties"]["batch_size"]
    assert bs["maximum"] == 16 and bs["default"] == 8
    assert "grad_accum" in bs["description"]


def test_max_label_tokens_within_eval_generation_budget():
    """A transcript allowed in training must fit eval's max_new_tokens for the longest (30 s) clip."""
    from mlops_backend.prepare.qwen_asr import MAX_SECONDS
    from mlops_backend.prompt import max_new_tokens

    mlt = ASR["schema"]["properties"]["hparams"]["properties"]["max_label_tokens"]
    assert mlt["maximum"] == max_new_tokens(MAX_SECONDS) == 916


def test_schema_accepts_and_rejects():
    v = Draft202012Validator(ASR["schema"])
    ok = {"base_model": MODEL, "method": {"type": "full"}, "hparams": {"batch_size": 16, "learning_rate": 1e-3},
          "export": ["safetensors"]}
    assert not list(v.iter_errors(ok))
    for bad in ({"hparams": {"batch_size": 17}}, {"hparams": {"max_label_tokens": 917}},
                {"hparams": {"learning_rate": 1e-2}},
                {"method": {"type": "lora"}}, {"export": ["ct2"]}, {"base_model": "Qwen/Qwen3-ASR-0.6B"}):
        assert list(v.iter_errors({**ok, **bad})), bad


def test_defaults_match_module():
    from mlops_backend.qwen_asr import DEFAULTS

    hp = ASR["schema"]["properties"]["hparams"]["properties"]
    for k, val in DEFAULTS.items():
        assert hp[k]["default"] == val, k


def test_catalog_recipe_validates_against_schema():
    from app.core.catalog import load_catalog

    entries, errors = load_catalog(REPO / "catalog")
    assert errors == []
    entry = {e.id: e for e in entries}["qwen3-asr-0-6b-bn"]
    tpl = entry.recipe_template
    assert entry.task == "asr" and entry.hf_repo == MODEL
    assert tpl.backend == "qwen-asr" and tpl.method == {"type": "full"}
    Draft202012Validator(ASR["schema"]).validate(
        {"base_model": entry.hf_repo, "method": tpl.method, "hparams": tpl.hparams, "export": tpl.export})


def _cfg(model):
    return RunConfig(
        run_id="r1", attempt=1, resume=False, task="asr", backend="qwen-asr", base_model=model,
        method={"type": "full"}, hparams={}, export=[], seed=1, datasets={}, prepared_root="prepared")


def test_phase_dispatch_unknown_model_fails(tmp_path):
    rd = RunDir(tmp_path / "run")
    rd.ensure()
    write_config(rd, _cfg("x/y"))
    r = subprocess.run([sys.executable, "-m", "mlops_backend", "train", "--run-dir", str(rd.path)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 1
    assert "unsupported base_model 'x/y' for backend qwen-asr" in r.stderr


def test_phase_not_implemented_is_clean(tmp_path, monkeypatch, capsys):
    from app.core.contract import Phase
    from mlops_backend import phases

    monkeypatch.setitem(phases._FAMILY_MODULE, "qwen-asr", "_no_such_module")
    rd = RunDir(tmp_path / "run")
    rd.ensure()
    assert phases._dispatch(Phase.train)(rd, _cfg(MODEL), None) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "support not installed" in err
