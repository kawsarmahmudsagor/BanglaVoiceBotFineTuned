# Reference layout restructure — design

Date: 2026-11-23. Branch: `refactor/reference-layout`.

## Goal

Make the repository follow the layout of the `MLops-Training-Pipeline` project: Python code under `backend/` (package `app`, a `workers/` directory), the Next.js app under `frontend/`. Behaviour does not change. Every test that passes today passes after the move.

## Constraints

- The project runs on a Linux VM. Lockfiles, Dockerfiles and scripts must work there. Windows `run-local.bat` keeps working for local development.
- Python stays at 3.12 (`>=3.12,<3.13`). The reference project's 3.13 is not adopted.
- Use `git mv` for every move so history follows the files.
- Old specs and plans under `docs/superpowers/` keep their old paths; they are a historical record.
- Real training backends keep their own venvs and must be able to import the backend contract without installing FastAPI, SQLAlchemy, pyarrow or other server dependencies.

## Target tree

```
backend/
  pyproject.toml        uv workspace root + project "bangla-voice-backend"
  uv.lock
  .python-version
  conftest.py
  Dockerfile            was api/Dockerfile
  alembic.ini           was api/alembic.ini
  alembic/              was api/alembic/
  app/                  was api/mlops_api/
    core/               was shared/mlops_shared/
  workers/              was worker/mlops_worker/
    backends/
      fake/             was backends/fake/ (workspace member, module mlops_backend)
  tests/
    app/                was api/tests/
    core/               was shared/tests/
    workers/            was worker/tests/
frontend/               was web/
catalog/  recipes/  deploy/  docs/  scripts/   unchanged location
docker-compose.yml  run-local.bat  README.md  .env.example   stay at root, paths updated
```

The root `pyproject.toml` and `uv.lock` are removed; `backend/` is the only uv workspace root.

## Packaging

`backend/pyproject.toml` declares one project, `bangla-voice-backend`, with wheel packages `app` and `workers`.

- Base dependencies: `pydantic>=2.7`, `pyyaml`. This is what `app.core` needs.
- Extra `validate`: `jsonschema>=4` (was `mlops-shared[validate]`).
- Extra `server`: the current `mlops-api` dependencies (FastAPI, uvicorn, SQLAlchemy, asyncpg, aiosqlite, alembic, pydantic-settings, pyarrow, numpy, soundfile, pyloudnorm, nvidia-ml-py, sse-starlette, python-multipart) plus `jsonschema>=4`.
- Extra `worker`: `mlflow-skinny>=2.14` plus everything in `server` (the worker imports `app.models`, `app.db`, `app.settings`).
- Workspace members: `workers/backends/fake`. Its project depends on `bangla-voice-backend` (workspace source).
- Dev group: unchanged (`pytest`, `pytest-asyncio`, `httpx`, `jsonschema`). The dev environment syncs with `--all-extras`.
- Real backends live at `backend/workers/backends/<name>/`, are listed in `exclude`, and depend on `bangla-voice-backend` by path (`../../..`, editable) with no extras.

`app/__init__.py` and `app/core/__init__.py` must not import anything beyond the standard library, so importing `app.core.contract` in a backend venv does not need server dependencies. A test enforces this by importing `app.core.contract` in a subprocess and asserting `fastapi` and `sqlalchemy` are absent from `sys.modules`.

## Import renames

| Old | New |
| --- | --- |
| `mlops_api` | `app` |
| `mlops_shared` | `app.core` |
| `mlops_worker` | `workers` |
| `mlops_backend` | unchanged |

The worker entry point becomes `python -m workers`. The API entry point becomes `app.app:create_app`.

## Path-dependent code

- `app/settings.py`: `REPO_ROOT` stays the repository root (`Path(__file__).resolve().parents[2]` still resolves to it). `.env` stays at the repository root. Default `backends_root` becomes `REPO_ROOT / "backend" / "workers" / "backends"`.
- `app/backends.py`: the workspace root is `backends_root.parents[1]` (that is, `backend/`), and the member path checked against `members`/`exclude` is `workers/backends/<name>`. The root venv is `backend/.venv`.
- Tests that compute the repo root from `__file__` are updated for the new depth.
- `scripts/setup_venvs.sh`: syncs `backend/` with `--all-extras`, then each `backend/workers/backends/*` except `fake`, checking the `exclude` list in `backend/pyproject.toml`.
- `scripts/dev.sh`: runs the worker with `uv --directory backend run python -m workers`.
- `run-local.bat`: `uv sync --frozen --all-extras` and all `uv run` calls in `backend/`, alembic with `backend/alembic.ini`, web from `frontend/`, `BACKENDS_ROOT=%ROOT%\backend\workers\backends`.

## Docker

- `backend/Dockerfile`, build context `./backend`. Copies `pyproject.toml uv.lock .python-version`, `app`, `workers`, `alembic*`. Runs `uv sync --frozen --no-dev --extra server`. CMD `uvicorn app.app:create_app --factory`.
- `docker-compose.yml`: api build context `./backend`; command runs `alembic -c alembic.ini upgrade head`; `BACKENDS_ROOT: ${REPO_ROOT}/backend/workers/backends`; volume `./backend/workers/backends:${REPO_ROOT}/backend/workers/backends:ro`. Web build context `./frontend`.
- `.dockerignore` files move with their directories; the root one is kept for `backend/`'s needs (`.venv`, `__pycache__`).

## Docs

`README.md`, `.env.example` and `.gitignore` are updated to the new paths and commands (`cd backend && uv run pytest`, `cd frontend && npm test`). `docs/context/*` is left alone.

## Verification

- `cd backend && uv lock` succeeds; `uv sync --all-extras` succeeds.
- `cd backend && uv run pytest` gives the same counts as before the move (208 passed, 10 skipped) plus the new import-isolation test.
- `cd frontend`: vitest (166), typecheck, lint and build pass.
- `docker compose config` parses.
- If Docker is running: `docker build ./backend` and `docker build ./frontend` succeed (Linux images).
- A grep for `mlops_api`, `mlops_shared`, `mlops_worker`, `api/`, `shared/`, `worker/` and `web/` outside `docs/superpowers` finds no stale references.

## Out of scope

Code changes beyond renames and paths; new features; Python version change; the 14 deferred minors from the catalog/UI work.
