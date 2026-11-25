import sys
from pathlib import Path

import pytest

from app.backends import BackendMissing, backend_python, load_capabilities
from app.settings import Settings

VENV_SUB = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"


def _repo(tmp_path: Path, members: list[str], root_venv: bool = True) -> Settings:
    ws = tmp_path / "repo" / "backend"
    (ws / "workers" / "backends" / "fake").mkdir(parents=True)
    quoted = ", ".join(f'"{m}"' for m in members)
    (ws / "pyproject.toml").write_text(f"[tool.uv.workspace]\nmembers = [{quoted}]\n", encoding="utf-8")
    if root_venv:
        py = ws / ".venv" / VENV_SUB
        py.parent.mkdir(parents=True)
        py.write_text("")
    return Settings(_env_file=None, data_root=tmp_path / "data", backends_root=ws / "workers" / "backends")


def test_load_capabilities_finds_fake(settings):
    assert load_capabilities(settings.backends_root)["fake"].tasks["llm-sft"].methods == ["none"]


def test_load_capabilities_skips_invalid(tmp_path):
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "capabilities.json").write_text("{nope")
    assert load_capabilities(tmp_path) == {}


def test_backend_python_missing_venv_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("BACKEND_PYTHON_FAKE", raising=False)
    s = _repo(tmp_path, members=["other"])
    with pytest.raises(BackendMissing):
        backend_python("fake", s)


def test_backend_python_workspace_member_uses_root_venv(tmp_path, monkeypatch):
    monkeypatch.delenv("BACKEND_PYTHON_FAKE", raising=False)
    s = _repo(tmp_path, members=["other", "workers/backends/fake"])
    assert backend_python("fake", s) == s.backends_root.parents[1] / ".venv" / VENV_SUB


def test_backend_python_own_venv_wins_over_root(tmp_path, monkeypatch):
    monkeypatch.delenv("BACKEND_PYTHON_FAKE", raising=False)
    s = _repo(tmp_path, members=["workers/backends/fake"])
    own = s.backends_root / "fake" / ".venv" / VENV_SUB
    own.parent.mkdir(parents=True)
    own.write_text("")
    assert backend_python("fake", s) == own


def test_backend_python_env_override(settings):
    assert backend_python("fake", settings) == Path(sys.executable)


def test_backend_python_env_override_hyphenated_name(settings, monkeypatch):
    monkeypatch.setenv("BACKEND_PYTHON_QWEN_ASR", "/x/python")
    assert backend_python("qwen-asr", settings) == Path("/x/python")


def test_backend_python_excluded_backend_is_not_a_member(tmp_path, monkeypatch):
    monkeypatch.delenv("BACKEND_PYTHON_FAKE", raising=False)
    s = _repo(tmp_path, members=["workers/backends/*"])
    with (s.backends_root.parents[1] / "pyproject.toml").open("a", encoding="utf-8") as f:
        f.write('exclude = ["workers/backends/fake"]\n')
    with pytest.raises(BackendMissing):
        backend_python("fake", s)
