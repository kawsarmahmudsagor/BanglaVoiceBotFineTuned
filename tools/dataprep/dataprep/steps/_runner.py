"""Shared resume logic for per-row steps.

A default rerun computes only ids not yet in the step file; rows with `error` set are kept and not retried;
`retry_errors` recomputes only errored rows; `force` recomputes all; ids no longer in clips.parquet are dropped.
"""
from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable

import pandas as pd

from ..pool import run_rows
from ..workdir import StepFile, WorkDir


def run_step(
    wd: WorkDir,
    *,
    step: str,
    version: int,
    columns: list[str],
    row_fn: Callable[[dict], dict],
    input_cols: list[str],
    dtypes: dict[str, str],
    workers: int,
    force: bool,
    retry_errors: bool,
    params: dict | None = None,
    only_ids: set[str] | None = None,
    rows_fn: Callable[[list[dict]], list[dict]] | None = None,
    checkpoint_every: int | None = None,
) -> int:
    """Run row_fn over the clips that still need it and rewrite the step file. Returns rows computed.

    `only_ids` restricts the clips to compute (rows outside it are simply not scored). `rows_fn` replaces
    row_fn/workers: it receives rows to compute and returns one result dict per row (batched steps).
    Rows are always processed in chunks and the step file is rewritten (merged with previous rows) after each
    chunk, so an interrupted run keeps finished chunks; a progress line (rows done/total, rows/s, ETA) goes to
    stderr after each chunk. The chunk is `checkpoint_every` when given, else max(4096, ceil(rows/50)).
    """
    params = params or {}
    clips = StepFile.read(wd.clips_path, step="import", version=1)
    path = wd.step_path(step)
    prev = pd.DataFrame(columns=columns)
    todo = clips
    if path.exists() and not force:
        prev = StepFile.read(path, step=step, version=version, params=params)
        prev = prev[prev["id"].isin(clips["id"])]  # drop ids no longer in clips.parquet
        if retry_errors:
            prev = prev[prev["error"].isna()]
        todo = clips[~clips["id"].isin(prev["id"])]
    if only_ids is not None:
        todo = todo[todo["id"].isin(only_ids)]
    todo_rows = todo[input_cols].to_dict("records")

    def compute(rows: list[dict]) -> list[dict]:
        return rows_fn(rows) if rows_fn else run_rows(row_fn, rows, workers=workers, desc=step)

    def write(parts: list[pd.DataFrame]) -> pd.DataFrame:
        new_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
        out = new_df if prev.empty else pd.concat([prev, new_df], ignore_index=True)
        StepFile.write(path, out.astype(dtypes), step=step, version=version, params=params)
        return new_df

    total = len(todo_rows)
    size = checkpoint_size(total, checkpoint_every)
    parts: list[pd.DataFrame] = []
    chunks = [todo_rows[i : i + size] for i in range(0, total, size)] or [[]]
    start, done = time.perf_counter(), 0
    for chunk in chunks:
        parts.append(pd.DataFrame(compute(chunk)).reindex(columns=columns))
        write(parts)
        done += len(chunk)
        if total:
            _progress(step, done, total, time.perf_counter() - start)
    return sum(len(p) for p in parts)


DEFAULT_MIN_CHUNK = 4096
MAX_CHECKPOINTS = 50


def checkpoint_size(total: int, checkpoint_every: int | None) -> int:
    """Rows per checkpoint: `checkpoint_every` when given, else max(4096, ceil(total/50)) (<= ~50 rewrites)."""
    if checkpoint_every:
        return checkpoint_every
    return max(DEFAULT_MIN_CHUNK, math.ceil(total / MAX_CHECKPOINTS))


def _progress(step: str, done: int, total: int, elapsed: float) -> None:
    rate = done / elapsed if elapsed > 0 else float("inf")
    eta = (total - done) / rate if rate not in (0, float("inf")) else 0.0
    m, s = divmod(int(round(eta)), 60)
    h, m = divmod(m, 60)
    print(f"{step}: {done}/{total} rows ({100 * done / total:.0f}%), {rate:.1f} rows/s, ETA {h:d}:{m:02d}:{s:02d}",
          file=sys.stderr, flush=True)
