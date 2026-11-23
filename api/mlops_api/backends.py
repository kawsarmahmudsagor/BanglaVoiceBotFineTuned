import logging
import os
import sys
from pathlib import Path

from mlops_shared.recipes import Capabilities

from .settings import Settings

log = logging.getLogger(__name__)


class BackendMissing(RuntimeError):
    pass


def load_capabilities(backends_root: Path) -> dict[str, Capabilities]:
    out: dict[str, Capabilities] = {}
    for path in sorted(Path(backends_root).glob("*/capabilities.json")):
        try:
            caps = Capabilities.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("skipping invalid capabilities file %s: %s", path, exc)
            continue
        out[caps.name] = caps
    return out


def backend_python(name: str, settings: Settings) -> Path:
    env = os.environ.get(f"BACKEND_PYTHON_{name.upper()}")
    if env:
        return Path(env)
    sub = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    p = settings.backends_root / name / ".venv" / sub
    if not p.exists():
        raise BackendMissing("backend venv missing; run scripts/setup_venvs.sh")
    return p
