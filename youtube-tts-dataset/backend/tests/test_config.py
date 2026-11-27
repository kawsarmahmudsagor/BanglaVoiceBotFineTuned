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
    assert t.min_bn_ratio == 0.5
    assert (t.min_chars_per_s, t.max_chars_per_s, t.max_word_repeat) == (3.0, 30.0, 6)
    assert t.dnsmos_min == 3.0
    assert t.speaker_sim_min == 0.60
    assert (t.clip_min_s, t.clip_max_s, t.demucs_segment_s) == (15.0, 25.0, 7.0)
    assert (t.piece_max_s, t.piece_gap_s, t.tail_min_s, t.disagree_cer, t.tie_margin) == (10.0, 0.3, 5.0, 0.25, 0.05)
    assert not hasattr(t, "merge_gap_s")


def test_export_hf_token_sets_only_when_missing(tmp_path):
    from ytds.config import Settings, export_hf_token

    env: dict[str, str] = {}
    export_hf_token(Settings(hf_token="from-dotenv"), env)
    assert env == {"HF_TOKEN": "from-dotenv"}

    env = {"HF_TOKEN": "from-process"}
    export_hf_token(Settings(hf_token="from-dotenv"), env)
    assert env == {"HF_TOKEN": "from-process"}

    env = {}
    export_hf_token(Settings(), env)
    assert env == {}


def test_create_app_exports_hf_token(tmp_path, monkeypatch):
    import os

    from ytds.api import create_app
    from ytds.config import Settings
    from ytds.store import Store

    monkeypatch.delenv("HF_TOKEN", raising=False)
    create_app(Settings(data_dir=tmp_path, hf_token="tok"), runner=object(), store=Store(tmp_path / "db"))
    assert os.environ["HF_TOKEN"] == "tok"


def test_model_paths_env_and_defaults(tmp_path):
    none = tmp_path / "none.env"
    s = load_settings(env={"MODEL_A_PATH": "x", "MODEL_B_PATH": "y"}, sub_env=none, root_env=none)
    assert (s.model_a_path, s.model_b_path) == ("x", "y")
    d = load_settings(env={}, sub_env=none, root_env=none)
    assert d.model_a_path == r"C:\Users\BS23-DESKTOP-00038\Downloads\qwen-3-asr-finetune-1\safetensors"
    assert d.model_b_path == r"C:\Users\BS23-DESKTOP-00038\Downloads\whisper-finetune-1\modelswhisper-v3-bn-version1"
