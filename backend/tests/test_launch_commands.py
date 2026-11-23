"""Launch scripts must ask uv for the extras each process imports, so a fresh clone works
without an earlier `setup_venvs.sh` run and new server/worker deps get installed on launch."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _lines(name: str, needle: str) -> list[str]:
    text = (REPO / name).read_text(encoding="utf-8")
    found = [ln for ln in text.splitlines() if needle in ln and "uv run" in ln and not ln.lstrip().startswith(("rem", "#"))]
    assert found, f"no uv run line with {needle!r} in {name}"
    return found


def test_worker_launch_installs_worker_extra():
    for name in ("scripts/dev.sh", "run-local.bat"):
        for ln in _lines(name, "-m workers"):
            assert "--extra worker" in ln, ln


def test_api_and_migration_launch_install_server_extra():
    for needle in ("uvicorn app.app:create_app", "alembic -c"):
        for ln in _lines("run-local.bat", needle):
            assert "--extra server" in ln, ln
