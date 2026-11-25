import logging
import os
import sys
import tomllib
from fnmatch import fnmatch
from pathlib import Path

from app.core.recipes import Capabilities

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


def _is_root_workspace_member(name: str, workspace_root: Path) -> bool:
    try:
        data = tomllib.loads((workspace_root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    workspace = data.get("tool", {}).get("uv", {}).get("workspace", {})
    rel = f"workers/backends/{name}"
    if any(fnmatch(rel, e.rstrip("/")) for e in workspace.get("exclude", [])):
        return False
    return any(fnmatch(rel, m.rstrip("/")) for m in workspace.get("members", []))


def backend_python(name: str, settings: Settings) -> Path:
    # a shell cannot export a name with "-": backend qwen-asr reads BACKEND_PYTHON_QWEN_ASR
    env = os.environ.get(f"BACKEND_PYTHON_{name.upper().replace('-', '_')}")
    if env:
        return Path(env)
    sub = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    p = settings.backends_root / name / ".venv" / sub
    if p.exists():
        return p
    # backends_root is <workspace>/workers/backends; the uv workspace root is backend/.
    workspace_root = settings.backends_root.parents[1]
    if _is_root_workspace_member(name, workspace_root):
        root_py = workspace_root / ".venv" / sub
        if root_py.exists():
            return root_py
    raise BackendMissing("backend venv missing; run scripts/setup_venvs.sh")
