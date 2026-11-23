import sys
from pathlib import Path

import pytest

from mlops_api.backends import BackendMissing, backend_python, load_capabilities


def test_load_capabilities_finds_fake(settings):
    assert load_capabilities(settings.backends_root)["fake"].tasks["llm-sft"].methods == ["none"]


def test_load_capabilities_skips_invalid(tmp_path):
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "capabilities.json").write_text("{nope")
    assert load_capabilities(tmp_path) == {}


def test_backend_python_missing_venv_raises(settings, monkeypatch):
    monkeypatch.delenv("BACKEND_PYTHON_FAKE")
    with pytest.raises(BackendMissing):
        backend_python("fake", settings)


def test_backend_python_env_override(settings):
    assert backend_python("fake", settings) == Path(sys.executable)
