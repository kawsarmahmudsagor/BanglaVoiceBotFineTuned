"""Work directory layout and versioned step files (spec §4)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

META_KEY = b"dataprep"


class StaleStepError(RuntimeError):
    """A step file was written by another step, version or parameter set."""


class WorkDir:
    def __init__(self, root: Path):
        self.root = Path(root)

    @property
    def clips_path(self) -> Path:
        return self.root / "clips.parquet"

    def step_path(self, name: str) -> Path:
        return self.root / f"{name}.parquet"

    def selection_path(self, profile: str) -> Path:
        return self.root / "selection" / f"{profile}.parquet"

    def export_dir(self, dataset: str) -> Path:
        return self.root / "export" / dataset


class StepFile:
    @staticmethod
    def write(path: Path, df: pd.DataFrame, *, step: str, version: int, params: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df, preserve_index=False)
        meta = {"step": step, "version": version, "params": params}
        table = table.replace_schema_metadata({**(table.schema.metadata or {}), META_KEY: json.dumps(meta).encode()})
        tmp = path.with_name(path.name + ".tmp")
        pq.write_table(table, tmp)
        tmp.replace(path)

    @staticmethod
    def _meta(path: Path) -> dict:
        raw = (pq.read_schema(path).metadata or {}).get(META_KEY)
        if raw is None:
            raise StaleStepError(f"{path} has no dataprep metadata; rerun the step")
        return json.loads(raw)

    @staticmethod
    def read(path: Path, *, step: str, version: int, params: dict | None = None) -> pd.DataFrame:
        meta = StepFile._meta(Path(path))
        if meta["step"] != step or meta["version"] != version:
            raise StaleStepError(
                f"{path} is step '{meta['step']}' v{meta['version']}, expected '{step}' v{version}; rerun with --force"
            )
        if params is not None and meta["params"] != params:
            raise StaleStepError(f"{path} was written with params {meta['params']}, now {params}; rerun with --force")
        return pd.read_parquet(path)

    @staticmethod
    def done_ids(path: Path) -> set[str]:
        df = pd.read_parquet(path, columns=["id", "error"])
        return set(df.loc[df["error"].isna(), "id"])
