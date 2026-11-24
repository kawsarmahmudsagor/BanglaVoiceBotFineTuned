"""Hard-link-or-copy helpers (disk budget: checkpoints, final/ and export/ share inodes where the filesystem allows).
Never write into a linked file in place: replace it (unlink + new file) instead."""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link src at dst (following symlinks, e.g. HF cache snapshot entries); copy when that is not possible
    (other filesystem, no hard-link support). An existing dst is replaced."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    real = Path(os.path.realpath(src))
    try:
        os.link(real, dst)
    except OSError:
        shutil.copyfile(real, dst)


def link_tree(src: Path, dst: Path, skip: tuple[str, ...] = ()) -> None:
    """link_or_copy every file below src into dst (same relative layout); names in `skip` are left out at any level."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if any(part in skip for part in rel.parts):
            continue
        if p.is_dir():
            (dst / rel).mkdir(parents=True, exist_ok=True)
        else:
            link_or_copy(p, dst / rel)
