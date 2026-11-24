"""Shared resume logic for per-row steps.

A default rerun computes only ids not yet in the step file; rows with `error` set are kept and not retried;
`retry_errors` recomputes only errored rows; `force` recomputes all; ids no longer in clips.parquet are dropped.
"""
from __future__ import annotations

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
    `checkpoint_every` processes rows in chunks of that size and rewrites the step file (merged with previous
    rows) after each chunk, so an interrupted run keeps finished chunks. None writes once at the end.
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

    size = checkpoint_every or len(todo_rows) or 1
    parts: list[pd.DataFrame] = []
    chunks = [todo_rows[i : i + size] for i in range(0, len(todo_rows), size)] or [[]]
    for chunk in chunks:
        parts.append(pd.DataFrame(compute(chunk)).reindex(columns=columns))
        if checkpoint_every:
            write(parts)
    if not checkpoint_every:
        write(parts)
    return sum(len(p) for p in parts)
