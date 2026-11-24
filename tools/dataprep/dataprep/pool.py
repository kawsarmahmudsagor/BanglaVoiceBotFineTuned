"""Per-row step runner: row failures are recorded, never raised."""
from __future__ import annotations

import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def _safe(fn: Callable[[dict], dict], row: dict) -> dict:
    try:
        out = dict(fn(row))
        out.setdefault("id", row["id"])
        out.setdefault("error", None)
        return out
    except Exception as e:  # noqa: BLE001 - a bad row must not stop the step
        return {"id": row["id"], "error": f"{type(e).__name__}: {e}"}


def run_rows(fn: Callable[[dict], dict], rows: list[dict], *, workers: int, desc: str) -> list[dict]:
    """Apply fn to every row. workers=1 runs inline; otherwise fn must be picklable."""
    if workers <= 1:
        out = [_safe(fn, r) for r in rows]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            out = list(pool.map(partial(_safe, fn), rows, chunksize=64))
    failed = sum(1 for o in out if o["error"])
    print(f"{desc}: {len(out) - failed} ok, {failed} failed", file=sys.stderr)
    return out
