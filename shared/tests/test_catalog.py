"""Catalog entry schema and loader."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from mlops_shared.catalog import catalog_summary, load_catalog


def valid_entry(**overrides) -> dict:
    e = {
        "id": "whisper-x",
        "task": "asr",
        "name": "Whisper X",
        "hf_repo": "org/whisper-x",
        "params": "809M",
        "role": "primary",
        "license": {"id": "mit", "commercial_ok": True, "url": "https://example.com/l"},
        "bangla": {"status": "finetuned", "evidence": "WER 11%"},
        "runtime": {"engine": "faster_whisper", "convert": "ct2", "streaming": False},
        "description": "A model.",
        "dataset": {
            "format": "asr",
            "required_fields": ["file", "text"],
            "optional_fields": ["speaker"],
            "example": "a.wav,আমি",
            "recommended": {"hours": "10-100"},
        },
        "finetune": {
            "methods": ["full"],
            "frameworks": ["transformers"],
            "targets": ["decoder"],
            "hyperparams": {"lr": 1e-5},
            "metrics": ["wer"],
            "pitfalls": ["forgetting"],
        },
        "recipe_template": {"method": {"type": "full"}, "hparams": {"lr": 1e-5}, "export": ["ct2"]},
        "sources": [{"title": "card", "url": "https://example.com"}],
        "verified": "2026-11-23",
    }
    e = copy.deepcopy(e)
    e.update(overrides)
    return e


def write(root: Path, section: str, stem: str, data) -> Path:
    d = root / section
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{stem}.yaml"
    p.write_text(data if isinstance(data, str) else yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return p


def test_valid_entry_loads(tmp_path):
    write(tmp_path, "stt", "whisper-x", valid_entry())
    entries, errors = load_catalog(tmp_path)
    assert errors == []
    assert [e.id for e in entries] == ["whisper-x"]


def test_unknown_field_rejected(tmp_path):
    write(tmp_path, "stt", "whisper-x", valid_entry(foo=1))
    entries, errors = load_catalog(tmp_path)
    assert entries == []
    assert len(errors) == 1
    assert errors[0].file == "stt/whisper-x.yaml"


def test_task_section_mismatch(tmp_path):
    write(tmp_path, "tts", "whisper-x", valid_entry())
    entries, errors = load_catalog(tmp_path)
    assert entries == []
    assert "section" in errors[0].message


def test_dataset_format_must_equal_task(tmp_path):
    e = valid_entry()
    e["dataset"]["format"] = "tts"
    write(tmp_path, "stt", "whisper-x", e)
    entries, errors = load_catalog(tmp_path)
    assert entries == [] and len(errors) == 1


def test_id_must_match_filename(tmp_path):
    write(tmp_path, "stt", "b", valid_entry(id="a"))
    entries, errors = load_catalog(tmp_path)
    assert entries == []
    assert "does not match file name" in errors[0].message


def test_non_slug_id_rejected(tmp_path):
    write(tmp_path, "stt", "Bad_Id", valid_entry(id="Bad_Id"))
    write(tmp_path, "stt", "a", valid_entry(id="a\n"))
    entries, errors = load_catalog(tmp_path)
    assert entries == []
    assert len(errors) == 2


def test_bad_yaml_reported_others_load(tmp_path):
    write(tmp_path, "stt", "broken", "id: [unclosed\n")
    write(tmp_path, "stt", "whisper-x", valid_entry())
    entries, errors = load_catalog(tmp_path)
    assert len(entries) == 1
    assert [e.file for e in errors] == ["stt/broken.yaml"]


def test_missing_root_is_empty(tmp_path):
    assert load_catalog(tmp_path / "nope") == ([], [])


def test_sorted_by_task_role_name(tmp_path):
    write(tmp_path, "stt", "b-bench", valid_entry(id="b-bench", name="A", role="benchmark"))
    write(tmp_path, "stt", "c-prim", valid_entry(id="c-prim", name="Z", role="primary"))
    write(tmp_path, "stt", "a-prim", valid_entry(id="a-prim", name="M", role="primary"))
    tts = valid_entry(id="t", task="tts", name="T")
    tts["dataset"]["format"] = "tts"
    write(tmp_path, "tts", "t", tts)
    llm = valid_entry(id="l", task="llm-sft", name="L", role="benchmark")
    llm["dataset"]["format"] = "llm-sft"
    write(tmp_path, "llm", "l", llm)
    entries, errors = load_catalog(tmp_path)
    assert errors == []
    assert [e.id for e in entries] == ["a-prim", "c-prim", "b-bench", "l", "t"]


def test_summary_fields(tmp_path):
    write(tmp_path, "stt", "whisper-x", valid_entry())
    (e,), _ = load_catalog(tmp_path)
    s = catalog_summary(e)
    assert set(s) == {"id", "task", "name", "hf_repo", "params", "role", "license", "bangla_status", "streaming"}
    assert s["bangla_status"] == "finetuned"
    assert s["streaming"] is False
    assert s["license"]["commercial_ok"] is True
