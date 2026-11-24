"""CosyVoice 3 export: `cosyvoice-dir` = final/ (hard-linked where possible) + refs/ + serve.json."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from app.core.backend_kit import export_all
from app.core.backend_kit.runutil import dir_artifact
from app.core.contract import RunConfig, RunDir

from .files import link_or_copy, link_tree  # noqa: F401  (link_or_copy re-exported)
from .train import prepared_dir

SERVE_WITH = "cosyvoice"


def _require_final(rd: RunDir) -> Path:
    final = rd.checkpoints_dir / "final"
    if not (final / ".complete").exists():
        raise RuntimeError(f"no completed final checkpoint at {final}")
    return final


def cosyvoice_exporters(rd: RunDir, cfg: RunConfig) -> dict:
    def cosyvoice_dir(out: Path):
        final = _require_final(rd)
        link_tree(final, out, skip=(".complete",))
        refs = prepared_dir(rd, "train") / "refs"
        if refs.is_dir():
            shutil.copytree(refs, out / "refs")
        (out / "serve.json").write_text(json.dumps({"serve_with": SERVE_WITH}), encoding="utf-8")
        return [dir_artifact(rd, "cosyvoice-dir", out, SERVE_WITH)]

    return {"cosyvoice-dir": cosyvoice_dir}


def export_cosyvoice(rd: RunDir, cfg: RunConfig, rep) -> int:
    return export_all(rd, cfg, rep, cosyvoice_exporters(rd, cfg))
