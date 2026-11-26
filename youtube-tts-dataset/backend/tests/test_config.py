from pathlib import Path

from ytds.config import REPO_ROOT, SUB_ROOT, Thresholds, load_settings


def _write(p: Path, text: str) -> Path:
    p.write_text(text, encoding="utf-8")
    return p


def test_env_wins_over_files(tmp_path):
    sub = _write(tmp_path / "sub.env", "LLM_MODEL=from-sub\n")
    root = _write(tmp_path / "root.env", "LLM_MODEL=from-root\n")
    s = load_settings({"LLM_MODEL": "from-env"}, sub_env=sub, root_env=root)
    assert s.llm_model == "from-env"


def test_sub_env_wins_over_root_env(tmp_path):
    sub = _write(tmp_path / "sub.env", "LLM_MODEL=from-sub\nYTDS_PORT=1111\n")
    root = _write(tmp_path / "root.env", "LLM_MODEL=from-root\nYTDS_PORT=2222\n")
    s = load_settings({}, sub_env=sub, root_env=root)
    assert s.llm_model == "from-sub"
    assert s.port == 1111


def test_root_env_used_when_sub_missing(tmp_path):
    root = _write(tmp_path / "root.env", "HF_TOKEN=tok\n")
    s = load_settings({}, sub_env=tmp_path / "nope.env", root_env=root)
    assert s.hf_token == "tok"


def test_defaults(tmp_path):
    s = load_settings({}, sub_env=tmp_path / "a.env", root_env=tmp_path / "b.env")
    assert s.llm_model == "gemma4:31b-cloud"
    assert s.port == 8765
    assert s.ollama_host == "http://localhost:11434"
    assert s.data_dir == SUB_ROOT / "var"
    assert s.export_dir == REPO_ROOT / "var" / "import"
    t = Thresholds()
    assert t.cer_auto == 0.05
    assert t.cer_fallback == 0.30
    assert t.dnsmos_min == 3.0
    assert t.speaker_sim_min == 0.60
    assert (t.clip_min_s, t.clip_max_s, t.merge_gap_s, t.demucs_segment_s) == (2.0, 15.0, 0.30, 7.0)
