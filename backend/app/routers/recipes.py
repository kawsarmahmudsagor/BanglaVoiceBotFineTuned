from __future__ import annotations

from pathlib import Path

import yaml
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.recipes import Capabilities, RecipeSpec, parse_dataset_ref, validate_recipe

from ..backends import load_capabilities
from ..runs_init import resolve_init
from ..db import get_session
from ..models import Dataset, DatasetVersion, Recipe, RecipeVersion
from .datasets import SLUG, iso

router = APIRouter(prefix="/api")

HEADER = "# Generated from the platform database. Edit in the UI.\n"


class RecipeBody(BaseModel):
    yaml: str | None = None
    spec: dict | None = None

    @model_validator(mode="after")
    def _one_of(self) -> "RecipeBody":
        if (self.yaml is None) == (self.spec is None):
            raise ValueError("provide exactly one of 'yaml' or 'spec'")
        return self


class CloneBody(BaseModel):
    new_name: str = Field(pattern=SLUG)


async def check_recipe(
    session: AsyncSession, spec: RecipeSpec, caps: dict[str, Capabilities], settings=None
) -> list[str]:
    errors = validate_recipe(spec, caps)
    if spec.backend not in caps:
        return errors
    refs = [("train", spec.datasets.train)]
    if spec.datasets.eval:
        refs.append(("eval", spec.datasets.eval))
    for slot, ref in refs:
        try:
            name, number = parse_dataset_ref(ref)
        except ValueError as e:
            errors.append(str(e))
            continue
        ds = (await session.execute(select(Dataset).where(Dataset.name == name))).scalar_one_or_none()
        found = ds is not None and (
            await session.execute(
                select(DatasetVersion.id).where(DatasetVersion.dataset_id == ds.id, DatasetVersion.number == number)
            )
        ).first() is not None
        if not found:
            errors.append(f"dataset '{ref}' not found")
            continue
        if ds.task != spec.task:
            errors.append(f"dataset '{name}' is task '{ds.task}', recipe is task '{spec.task}'")
        if slot == "train" and ds.eval_only:
            errors.append(f"dataset '{name}' is eval-only and cannot be used for training")
    errors.extend((await resolve_init(session, spec, settings))[1])
    return errors


def _parse(body: RecipeBody) -> tuple[RecipeSpec | None, list[str]]:
    data = body.spec
    if body.yaml is not None:
        try:
            data = yaml.safe_load(body.yaml)
        except yaml.YAMLError as e:
            return None, [f"yaml: {e}"]
        if not isinstance(data, dict):
            return None, ["yaml: top level must be a mapping"]
    try:
        return RecipeSpec.model_validate(data), []
    except ValidationError as e:
        return None, [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()]


async def _parse_and_check(
    request: Request, session: AsyncSession, body: RecipeBody
) -> tuple[RecipeSpec | None, list[str]]:
    spec, errors = _parse(body)
    if spec is None:
        return None, errors
    caps = load_capabilities(request.app.state.settings.backends_root)
    return spec, await check_recipe(session, spec, caps, request.app.state.settings)


def _dump(spec: RecipeSpec) -> tuple[dict, str]:
    data = spec.model_dump(mode="json", by_alias=True)
    return data, yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def _write_file(request: Request, name: str, text: str) -> None:
    d: Path = request.app.state.settings.recipes_dir
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.yaml").write_text(HEADER + text, encoding="utf-8")


async def _recipe(session: AsyncSession, name: str) -> Recipe:
    r = (await session.execute(select(Recipe).where(Recipe.name == name))).scalar_one_or_none()
    if r is None:
        raise HTTPException(404, f"recipe '{name}' not found")
    return r


async def _versions(session: AsyncSession, r: Recipe) -> list[RecipeVersion]:
    q = select(RecipeVersion).where(RecipeVersion.recipe_id == r.id).order_by(RecipeVersion.number)
    return list((await session.execute(q)).scalars())


@router.get("/backends")
async def list_backends(request: Request) -> list[dict]:
    caps = load_capabilities(request.app.state.settings.backends_root)
    return [c.model_dump(by_alias=True) for c in caps.values()]


@router.post("/recipes/validate")
async def validate(body: RecipeBody, request: Request, session: AsyncSession = Depends(get_session)) -> dict:
    spec, errors = await _parse_and_check(request, session, body)
    return {"errors": errors, "spec": spec.model_dump(mode="json", by_alias=True) if spec else None}


@router.post("/recipes", status_code=201, response_model=None)
async def save_recipe(body: RecipeBody, request: Request, session: AsyncSession = Depends(get_session)):
    spec, errors = await _parse_and_check(request, session, body)
    if errors or spec is None:
        return JSONResponse({"errors": errors}, status_code=422)
    data, text = _dump(spec)
    recipe = (await session.execute(select(Recipe).where(Recipe.name == spec.name))).scalar_one_or_none()
    if recipe is None:
        recipe = Recipe(name=spec.name)
        session.add(recipe)
        await session.flush()
    versions = await _versions(session, recipe)
    if versions and not versions[-1].frozen:
        v = versions[-1]
        v.spec, v.yaml = data, text
    else:
        v = RecipeVersion(recipe_id=recipe.id, number=(versions[-1].number + 1) if versions else 1,
                          spec=data, yaml=text, frozen=False)
        session.add(v)
    await session.commit()
    _write_file(request, spec.name, text)
    return {"name": spec.name, "number": v.number}


@router.get("/recipes")
async def list_recipes(session: AsyncSession = Depends(get_session)) -> list[dict]:
    out = []
    for r in (await session.execute(select(Recipe).order_by(Recipe.name))).scalars():
        versions = await _versions(session, r)
        if not versions:
            continue
        v = versions[-1]
        out.append({"name": r.name, "latest": {
            "number": v.number, "frozen": v.frozen, "task": v.spec["task"], "backend": v.spec["backend"],
            "base_model": v.spec["base_model"], "updated_at": iso(v.created_at)}})
    return out


@router.get("/recipes/{name}")
async def get_recipe(name: str, session: AsyncSession = Depends(get_session)) -> dict:
    r = await _recipe(session, name)
    return {"name": r.name, "versions": [
        {"number": v.number, "frozen": v.frozen, "created_at": iso(v.created_at)}
        for v in await _versions(session, r)]}


@router.get("/recipes/{name}/versions/{n}")
async def get_recipe_version(name: str, n: int, session: AsyncSession = Depends(get_session)) -> dict:
    r = await _recipe(session, name)
    v = next((v for v in await _versions(session, r) if v.number == n), None)
    if v is None:
        raise HTTPException(404, f"version {n} of '{name}' not found")
    return {"name": r.name, "number": v.number, "frozen": v.frozen, "spec": v.spec, "yaml": v.yaml}


@router.post("/recipes/{name}/clone", status_code=201)
async def clone_recipe(
    name: str, body: CloneBody, request: Request, session: AsyncSession = Depends(get_session)
) -> dict:
    src = await _recipe(session, name)
    versions = await _versions(session, src)
    exists = (await session.execute(select(Recipe.id).where(Recipe.name == body.new_name))).first()
    if exists:
        raise HTTPException(409, f"recipe '{body.new_name}' already exists")
    spec = RecipeSpec.model_validate({**versions[-1].spec, "name": body.new_name})
    data, text = _dump(spec)
    new = Recipe(name=body.new_name)
    session.add(new)
    await session.flush()
    session.add(RecipeVersion(recipe_id=new.id, number=1, spec=data, yaml=text, frozen=False))
    await session.commit()
    _write_file(request, body.new_name, text)
    return {"name": body.new_name, "number": 1}
