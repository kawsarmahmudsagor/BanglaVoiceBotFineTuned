"""Model catalog: one YAML file per fine-tunable base model, validated by `CatalogEntry`."""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.core.recipes import _SLUG_RE

SECTION_TASK: dict[str, str] = {"llm": "llm-sft", "stt": "asr", "tts": "tts"}
_ROLE_ORDER = {"primary": 0, "comparison": 1, "benchmark": 2}

TaskName = Literal["llm-sft", "asr", "tts"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class License(_Strict):
    id: str
    commercial_ok: bool
    url: str
    notes: str | None = None


class Bangla(_Strict):
    status: Literal["native", "finetuned", "related", "none"]
    evidence: str
    url: str | None = None


class Runtime(_Strict):
    engine: str
    convert: str | None = None
    streaming: bool


class DatasetGuide(_Strict):
    format: TaskName
    required_fields: list[str]
    optional_fields: list[str] = []
    example: str
    recommended: dict[str, str]
    notes: str | None = None


class FinetuneGuide(_Strict):
    methods: list[str] = Field(min_length=1)
    frameworks: list[str]
    targets: list[str]
    hyperparams: dict
    metrics: list[str]
    pitfalls: list[str]


class RecipeTemplate(_Strict):
    backend: str | None = None
    method: dict
    hparams: dict
    export: list[str] = []

    @field_validator("method")
    @classmethod
    def _method_has_type(cls, v: dict) -> dict:
        if "type" not in v:
            raise ValueError("method must contain 'type'")
        return v


class Source(_Strict):
    title: str
    url: str


class CatalogEntry(_Strict):
    id: str
    task: TaskName
    name: str
    hf_repo: str
    derived_from: str | None = None
    params: str
    role: Literal["primary", "comparison", "benchmark"]
    license: License
    bangla: Bangla
    runtime: Runtime
    description: str
    dataset: DatasetGuide
    finetune: FinetuneGuide
    recipe_template: RecipeTemplate
    sources: list[Source] = Field(min_length=1)
    verified: datetime.date

    @field_validator("id")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not _SLUG_RE.fullmatch(v):
            raise ValueError("id must be a kebab-case slug")
        return v

    @model_validator(mode="after")
    def _format_matches_task(self) -> CatalogEntry:
        if self.dataset.format != self.task:
            raise ValueError(f"dataset.format '{self.dataset.format}' must equal task '{self.task}'")
        return self


class CatalogError(BaseModel):
    file: str
    message: str


def load_catalog(root: Path) -> tuple[list[CatalogEntry], list[CatalogError]]:
    """Load every `root/<section>/*.yaml`. Bad files become errors; the rest still load."""
    entries: list[CatalogEntry] = []
    errors: list[CatalogError] = []
    if not root.is_dir():
        return entries, errors
    for section, task in SECTION_TASK.items():
        for path in sorted((root / section).glob("*.yaml")):
            rel = path.relative_to(root).as_posix()
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                entry = CatalogEntry.model_validate(data)
            except (yaml.YAMLError, ValidationError, UnicodeDecodeError) as exc:
                errors.append(CatalogError(file=rel, message=str(exc)))
                continue
            if entry.id != path.stem:
                errors.append(CatalogError(file=rel, message=f"id '{entry.id}' does not match file name '{path.stem}'"))
            elif entry.task != task:
                errors.append(CatalogError(file=rel, message=f"task '{entry.task}' does not belong in section '{section}'"))
            else:
                entries.append(entry)
    entries.sort(key=lambda e: (e.task, _ROLE_ORDER[e.role], e.name))
    return entries, errors


def catalog_summary(e: CatalogEntry) -> dict:
    return {
        "id": e.id,
        "task": e.task,
        "name": e.name,
        "hf_repo": e.hf_repo,
        "params": e.params,
        "role": e.role,
        "license": e.license.model_dump(mode="json"),
        "bangla_status": e.bangla.status,
        "streaming": e.runtime.streaming,
    }
