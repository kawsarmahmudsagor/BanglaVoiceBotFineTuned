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
) -> int:
    """Run row_fn over the clips that still need it and rewrite the step file. Returns rows computed.

    `only_ids` restricts the clips to compute (rows outside it are simply not scored).
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
    new = run_rows(row_fn, todo[input_cols].to_dict("records"), workers=workers, desc=step)
    new_df = pd.DataFrame(new).reindex(columns=columns)
    out = new_df if prev.empty else pd.concat([prev, new_df], ignore_index=True)
    out = out.astype(dtypes)
    StepFile.write(path, out, step=step, version=version, params=params)
    return len(new_df)
