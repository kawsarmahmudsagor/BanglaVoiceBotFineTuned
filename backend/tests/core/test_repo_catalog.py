"""Every catalog file committed to the repo validates."""
import json
from pathlib import Path

import jsonschema

from app.core.catalog import load_catalog

REPO = Path(__file__).resolve().parents[3]
BACKENDS = REPO / "backend" / "workers" / "backends"

# spec §2.1: catalog id -> backend of its recipe template
IN_SCOPE = {
    "qwen3-4b-instruct-2507": "unsloth",
    "gemma-3-4b-it": "unsloth",
    "titulm": "unsloth",
    "whisper-large-v3-turbo": "hf",
    "whisper-large-v3-turbo-bn": "hf",
    "wav2vec2-xls-r-300m": "hf",
    "mms-1b-all": "hf",
    "indic-parler-tts": "hf",
    "indicconformer": "nemo",
    "bangla-conformer": "nemo",
    "indicf5": "f5",
    "cosyvoice3": "cosyvoice",
    "qwen3-asr-0-6b-bn": "qwen-asr",
}


def test_repo_catalog_valid():
    entries, errors = load_catalog(REPO / "catalog")
    assert errors == []
    assert len(entries) >= 1


def test_recipe_templates_validate_against_backend_schemas():
    entries, errors = load_catalog(REPO / "catalog")
    assert errors == []
    by_id = {e.id: e for e in entries}
    for cid, backend in IN_SCOPE.items():
        assert cid in by_id, f"missing catalog entry {cid}"
        assert by_id[cid].recipe_template.backend == backend, cid
    for entry in entries:
        tpl = entry.recipe_template
        if cid_backend := IN_SCOPE.get(entry.id):
            assert tpl.backend == cid_backend
        else:
            assert tpl.backend is None, f"{entry.id} is out of scope; backend must be null"
            continue
        caps = json.loads((BACKENDS / tpl.backend / "capabilities.json").read_text(encoding="utf-8"))
        task = caps["tasks"][entry.task]
        assert tpl.method["type"] in task["methods"], entry.id
        assert tpl.export, f"{entry.id}: export must not be empty"
        for fmt in tpl.export:
            assert fmt in task["export_formats"], f"{entry.id}: export {fmt}"
        doc = {
            "base_model": entry.hf_repo,
            "method": tpl.method,
            "hparams": tpl.hparams,
            "export": tpl.export,
        }
        jsonschema.validate(doc, task["schema"])
        # template hparams use only names the backend schema knows
        known = set(task["schema"]["properties"]["hparams"]["properties"])
        assert set(tpl.hparams) <= known, f"{entry.id}: unknown hparams {set(tpl.hparams) - known}"


def test_templates_set_epochs_or_max_steps_not_both():
    """max_steps overrides epochs; a template with both hides which one decides the run length."""
    entries, _ = load_catalog(REPO / "catalog")
    for entry in entries:
        hp = entry.recipe_template.hparams
        assert not ("epochs" in hp and "max_steps" in hp), entry.id
