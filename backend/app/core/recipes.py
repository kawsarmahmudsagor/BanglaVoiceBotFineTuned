"""Recipe schema and validation against backend capabilities."""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .splits import SplitConfig

TASKS = ("llm-sft", "asr", "tts")

_SLUG_RE = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")  # used with fullmatch: "$" would accept a trailing newline
_REF_RE = re.compile(r"([a-z0-9]+(?:-[a-z0-9]+)*)@v([1-9][0-9]*)")


class BaseModelMeta(BaseModel):
    license: str
    commercial_ok: bool


class RecipeDatasets(BaseModel):
    train: str
    eval: str | None = None
    split: SplitConfig = SplitConfig()


class RecipeSpec(BaseModel):
    name: str
    task: Literal["llm-sft", "asr", "tts"]
    backend: str
    base_model: str
    base_model_meta: BaseModelMeta
    datasets: RecipeDatasets
    method: dict
    hparams: dict
    export: list[str] = []

    @field_validator("name")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not _SLUG_RE.fullmatch(v):
            raise ValueError("name must be a kebab-case slug")
        return v

    @field_validator("method")
    @classmethod
    def _method_has_type(cls, v: dict) -> dict:
        if "type" not in v:
            raise ValueError("method must contain 'type'")
        return v

    @property
    def seed(self) -> int:
        return self.hparams.get("seed", 42)


class TaskCapability(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    methods: list[str]
    export_formats: list[str]
    serve_formats: list[str]
    schema_: dict = Field(alias="schema")


class Capabilities(BaseModel):
    name: str
    label: str
    tasks: dict[str, TaskCapability]


def parse_dataset_ref(ref: str) -> tuple[str, int]:
    m = _REF_RE.fullmatch(ref)
    if not m:
        raise ValueError(f"invalid dataset ref {ref!r}; expected 'name@vN'")
    return m.group(1), int(m.group(2))


def validate_recipe(spec: RecipeSpec, caps: dict[str, Capabilities]) -> list[str]:
    backend = caps.get(spec.backend)
    if backend is None:
        return [f"unknown backend '{spec.backend}'"]
    task = backend.tasks.get(spec.task)
    if task is None:
        return [f"task '{spec.task}' not supported by backend '{spec.backend}'"]

    errors: list[str] = []
    mtype = spec.method["type"]
    if mtype not in task.methods:
        errors.append(f"method '{mtype}' not supported")
    for fmt in spec.export:
        name = fmt.split(":", 1)[0]
        if name not in task.export_formats:
            errors.append(f"export format '{name}' not supported")

    from jsonschema import Draft202012Validator  # lazy: optional extra "validate"

    validator = Draft202012Validator(task.schema_)
    instance = {"method": spec.method, "hparams": spec.hparams}
    for err in sorted(validator.iter_errors(instance), key=lambda e: list(map(str, e.absolute_path))):
        path = ".".join(str(p) for p in err.absolute_path)
        errors.append(f"{path}: {err.message}")
    return errors


def commercial_ok(base: BaseModelMeta, dataset_flags: Iterable[bool]) -> bool:
    return base.commercial_ok and all(dataset_flags)
