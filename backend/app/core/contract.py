"""Run-directory contract shared by the API, worker and backends."""
from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from .splits import SplitConfig


class Phase(str, Enum):
    prepare = "prepare"
    train = "train"
    eval = "eval"
    export = "export"


class DatasetRef(BaseModel):
    name: str
    version: int
    hash: str
    root: str
    manifest: str
    splits: list[str]
    split: SplitConfig | None = None


class RunConfig(BaseModel):
    run_id: str
    attempt: int
    resume: bool
    task: str
    backend: str
    base_model: str
    method: dict
    hparams: dict
    export: list[str]
    seed: int
    datasets: dict[str, DatasetRef]
    prepared_root: str


class Status(BaseModel):
    phase: Phase
    step: int = 0
    total_steps: int | None = None
    eta_s: float | None = None
    message: str = ""
    updated_at: float


class Metric(BaseModel):
    step: int
    split: str
    name: str
    value: float
    ts: float


class ExportArtifact(BaseModel):
    format: str
    path: str
    size_bytes: int
    serve_with: str | None = None


class ExportManifest(BaseModel):
    artifacts: list[ExportArtifact]


class EvalResults(BaseModel):
    metrics: dict[str, float]
    samples_dir: str | None = None


class PreparedRef(BaseModel):
    """One entry of prepared_refs.json: role plus PrepInfo fields, flat (kept as dicts to avoid importing prepare)."""

    model_config = ConfigDict(extra="allow")
    role: str


class RunDir:
    def __init__(self, path: Path):
        self.path = Path(path)

    @property
    def config_path(self) -> Path:
        return self.path / "config.json"

    @property
    def status_path(self) -> Path:
        return self.path / "status.json"

    @property
    def metrics_path(self) -> Path:
        return self.path / "metrics.jsonl"

    @property
    def logs_dir(self) -> Path:
        return self.path / "logs"

    @property
    def checkpoints_dir(self) -> Path:
        return self.path / "checkpoints"

    @property
    def eval_dir(self) -> Path:
        return self.path / "eval"

    @property
    def export_dir(self) -> Path:
        return self.path / "export"

    @property
    def eval_results_path(self) -> Path:
        return self.eval_dir / "results.json"

    @property
    def export_manifest_path(self) -> Path:
        return self.export_dir / "manifest.json"

    @property
    def eval_samples_path(self) -> Path:
        return self.eval_dir / "samples.jsonl"

    @property
    def eval_audio_dir(self) -> Path:
        return self.eval_dir / "audio"

    @property
    def prepared_refs_path(self) -> Path:
        return self.path / "prepared_refs.json"

    def ensure(self) -> None:
        for d in (self.path, self.logs_dir, self.checkpoints_dir, self.eval_dir, self.export_dir):
            d.mkdir(parents=True, exist_ok=True)


M = TypeVar("M", bound=BaseModel)


def _atomic_write(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(model.model_dump_json(indent=2).encode("utf-8"))
    os.replace(tmp, path)


def _read_model(path: Path, cls: type[M]) -> M | None:
    try:
        return cls.model_validate_json(path.read_bytes())
    except (FileNotFoundError, ValidationError, json.JSONDecodeError):
        return None


def write_config(rd: RunDir, cfg: RunConfig) -> None:
    _atomic_write(rd.config_path, cfg)


def read_config(rd: RunDir) -> RunConfig:
    return RunConfig.model_validate_json(rd.config_path.read_bytes())


def write_status(rd: RunDir, st: Status) -> None:
    _atomic_write(rd.status_path, st)


def read_status(rd: RunDir) -> Status | None:
    return _read_model(rd.status_path, Status)


def append_metric(rd: RunDir, m: Metric) -> None:
    rd.path.mkdir(parents=True, exist_ok=True)
    with rd.metrics_path.open("ab") as f:
        f.write(m.model_dump_json().encode("utf-8") + b"\n")


def read_metrics(rd: RunDir, offset: int = 0) -> tuple[list[Metric], int]:
    try:
        with rd.metrics_path.open("rb") as f:
            f.seek(offset)
            data = f.read()
    except FileNotFoundError:
        return [], offset
    end = data.rfind(b"\n") + 1  # only complete lines
    metrics: list[Metric] = []
    for line in data[:end].splitlines():
        if not line.strip():
            continue
        try:
            metrics.append(Metric.model_validate_json(line))
        except (ValidationError, json.JSONDecodeError):
            continue
    return metrics, offset + end


def write_eval_results(rd: RunDir, res: EvalResults) -> None:
    _atomic_write(rd.eval_results_path, res)


def read_eval_results(rd: RunDir) -> EvalResults | None:
    return _read_model(rd.eval_results_path, EvalResults)


def write_export_manifest(rd: RunDir, man: ExportManifest) -> None:
    _atomic_write(rd.export_manifest_path, man)


def read_export_manifest(rd: RunDir) -> ExportManifest | None:
    return _read_model(rd.export_manifest_path, ExportManifest)


def write_prepared_refs(rd: RunDir, refs: list[dict]) -> None:
    rd.prepared_refs_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = rd.prepared_refs_path.with_name(rd.prepared_refs_path.name + ".tmp")
    tmp.write_bytes(json.dumps(refs, indent=2, ensure_ascii=False).encode("utf-8"))
    os.replace(tmp, rd.prepared_refs_path)


def read_prepared_refs(rd: RunDir) -> list[dict]:
    try:
        data = json.loads(rd.prepared_refs_path.read_bytes())
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return []
    if not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
        return []
    return data


def append_eval_sample(rd: RunDir, sample: dict) -> None:
    rd.eval_dir.mkdir(parents=True, exist_ok=True)
    with rd.eval_samples_path.open("ab") as f:
        f.write(json.dumps(sample, ensure_ascii=False).encode("utf-8") + b"\n")


def read_eval_samples(rd: RunDir, offset: int, limit: int) -> tuple[list[dict], int]:
    """Return rows [offset, offset+limit) and the total number of valid rows."""
    try:
        data = rd.eval_samples_path.read_bytes()
    except FileNotFoundError:
        return [], 0
    rows: list[dict] = []
    for line in data[: data.rfind(b"\n") + 1].splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows[offset : offset + limit], len(rows)


def reset_eval_samples(rd: RunDir) -> None:
    """Drop samples.jsonl and eval/audio/*.wav so a re-run eval does not duplicate or leave stale samples."""
    rd.eval_samples_path.unlink(missing_ok=True)
    if rd.eval_audio_dir.is_dir():
        for p in rd.eval_audio_dir.glob("*.wav"):
            p.unlink(missing_ok=True)
