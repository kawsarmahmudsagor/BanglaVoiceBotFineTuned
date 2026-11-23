import subprocess
import sys

HEAVY = ("fastapi", "sqlalchemy", "pyarrow", "mlflow")


def test_core_imports_without_server_deps():
    """Real backends install bangla-voice-backend with no extras; app.core must not pull in server deps."""
    code = (
        "import sys, app.core.contract, app.core.recipes, app.core.catalog; "
        f"print(sorted(m for m in {HEAVY!r} if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]"
