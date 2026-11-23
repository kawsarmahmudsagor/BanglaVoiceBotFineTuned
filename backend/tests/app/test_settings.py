from pathlib import Path

from app.settings import Settings

REPO = Path(__file__).resolve().parents[3]


def test_env_file_is_repo_root_dotenv_not_cwd():
    assert Path(Settings.model_config["env_file"]) == REPO / ".env"


def test_env_file_values_load_and_unknown_keys_are_ignored(tmp_path, monkeypatch):
    monkeypatch.delenv("DISK_MIN_GB", raising=False)
    f = tmp_path / ".env"
    f.write_text("DISK_MIN_GB=5\nPOSTGRES_USER=mlops\nREPO_ROOT=/x\n", encoding="utf-8")
    assert Settings(_env_file=f).disk_min_gb == 5


def test_env_vars_beat_env_file(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("DISK_MIN_GB=5\n", encoding="utf-8")
    monkeypatch.setenv("DISK_MIN_GB", "7")
    assert Settings(_env_file=f).disk_min_gb == 7


def test_settings_fixture_ignores_a_developer_env_file(tmp_path, monkeypatch, request):
    f = tmp_path / ".env"
    f.write_text("WEB_ORIGIN=http://from-dotenv\n", encoding="utf-8")
    monkeypatch.setitem(Settings.model_config, "env_file", f)
    assert Settings().web_origin == "http://from-dotenv"  # the patch is live
    assert request.getfixturevalue("settings").web_origin == "http://localhost:3000"
