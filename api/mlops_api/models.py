import datetime as dt

from sqlalchemy import JSON, BigInteger, DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _ts(**kw):
    return mapped_column(DateTime(timezone=True), **kw)


class Upload(Base):
    __tablename__ = "upload"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    filename: Mapped[str] = mapped_column(String)
    length: Mapped[int] = mapped_column(BigInteger)
    state: Mapped[str] = mapped_column(String, default="uploading")  # uploading|complete
    created_at: Mapped[dt.datetime] = _ts(default=utcnow)


class Dataset(Base):
    __tablename__ = "dataset"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    task: Mapped[str] = mapped_column(String)
    eval_only: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[dt.datetime] = _ts(default=utcnow)


class DatasetVersion(Base):
    __tablename__ = "dataset_version"
    __table_args__ = (UniqueConstraint("dataset_id", "number"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id"))
    number: Mapped[int] = mapped_column()
    hash: Mapped[str] = mapped_column(String)
    path: Mapped[str] = mapped_column(String)
    n_samples: Mapped[int] = mapped_column()
    report: Mapped[dict] = mapped_column(JSON, default=dict)
    provenance: Mapped[dict] = mapped_column(JSON, default=dict)
    split_ratios: Mapped[dict] = mapped_column(JSON, default=dict)
    parent_version_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_version.id"), nullable=True)
    created_at: Mapped[dt.datetime] = _ts(default=utcnow)


class Ingest(Base):
    __tablename__ = "ingest"
    id: Mapped[int] = mapped_column(primary_key=True)
    upload_id: Mapped[str] = mapped_column(ForeignKey("upload.id"))
    dataset_name: Mapped[str] = mapped_column(String)
    task: Mapped[str] = mapped_column(String)
    eval_only: Mapped[bool] = mapped_column(default=False)
    provenance: Mapped[dict] = mapped_column(JSON, default=dict)
    state: Mapped[str] = mapped_column(String, default="pending")  # pending|running|done|failed
    report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    version_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_version.id"), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = _ts(default=utcnow)


class Recipe(Base):
    __tablename__ = "recipe"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    created_at: Mapped[dt.datetime] = _ts(default=utcnow)


class RecipeVersion(Base):
    __tablename__ = "recipe_version"
    __table_args__ = (UniqueConstraint("recipe_id", "number"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    recipe_id: Mapped[int] = mapped_column(ForeignKey("recipe.id"))
    number: Mapped[int] = mapped_column()
    spec: Mapped[dict] = mapped_column(JSON)
    yaml: Mapped[str] = mapped_column(Text)
    frozen: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[dt.datetime] = _ts(default=utcnow)


class Run(Base):
    __tablename__ = "run"
    id: Mapped[str] = mapped_column(String, primary_key=True)  # r-<8 hex>
    recipe_version_id: Mapped[int] = mapped_column(ForeignKey("recipe_version.id"))
    state: Mapped[str] = mapped_column(String, default="queued")
    attempt: Mapped[int] = mapped_column(default=1)
    resume: Mapped[bool] = mapped_column(default=False)
    cancel_requested: Mapped[bool] = mapped_column(default=False)
    run_dir: Mapped[str] = mapped_column(String)
    commercial_ok: Mapped[bool] = mapped_column(default=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    oom: Mapped[bool] = mapped_column(default=False)
    git_sha: Mapped[str | None] = mapped_column(String, nullable=True)
    lock_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    dataset_hashes: Mapped[dict] = mapped_column(JSON, default=dict)
    mlflow_run_id: Mapped[str | None] = mapped_column(String, nullable=True)
    started_at: Mapped[dt.datetime | None] = _ts(nullable=True)
    finished_at: Mapped[dt.datetime | None] = _ts(nullable=True)
    created_at: Mapped[dt.datetime] = _ts(default=utcnow)


class ModelVersion(Base):
    __tablename__ = "model_version"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String)
    number: Mapped[int] = mapped_column()
    run_id: Mapped[str] = mapped_column(ForeignKey("run.id"), unique=True)
    task: Mapped[str] = mapped_column(String)
    base_model: Mapped[str] = mapped_column(String)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    exports: Mapped[dict] = mapped_column(JSON, default=dict)
    commercial_ok: Mapped[bool] = mapped_column(default=False)
    champion: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[dt.datetime] = _ts(default=utcnow)


class GpuLock(Base):
    __tablename__ = "gpu_lock"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)
    holder_kind: Mapped[str | None] = mapped_column(String, nullable=True)
    holder_id: Mapped[str | None] = mapped_column(String, nullable=True)
    acquired_at: Mapped[dt.datetime | None] = _ts(nullable=True)
    created_at: Mapped[dt.datetime] = _ts(default=utcnow)
