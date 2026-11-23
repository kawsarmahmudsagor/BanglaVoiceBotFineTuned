# Reference Layout Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the repository to the `backend/` + `frontend/` layout of the MLops-Training-Pipeline project, with Python imports renamed to `app`, `app.core` and `workers`, and no change in behaviour.

**Architecture:** `backend/` becomes the only uv workspace root and one project (`bangla-voice-backend`, wheel packages `app` and `workers`) with a light base install and `server`/`worker` extras. The fake backend moves to `backend/workers/backends/fake/` and stays a workspace member. `web/` becomes `frontend/`. Root files (`docker-compose.yml`, `run-local.bat`, `README.md`, `.env.example`) stay at the root with updated paths.

**Tech Stack:** uv workspaces, hatchling, pytest, FastAPI, Alembic, Next.js 15, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-11-23-reference-layout-design.md`

## Global Constraints

- Target is a Linux VM: lockfile, Dockerfiles and shell scripts must work there; `run-local.bat` keeps working on Windows.
- Python stays `>=3.12,<3.13`; `.python-version` stays `3.12`.
- Every move uses `git mv` (history follows). Stage files by explicit path, never `git add -A`. Never `--no-verify`.
- Do not edit anything under `docs/superpowers/` except this plan's own ledger, and do not edit `docs/context/`.
- Import renames, applied with word boundaries in this order: `mlops_shared` → `app.core`, `mlops_api` → `app`, `mlops_worker` → `workers`. `mlops_backend` is unchanged.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Web tests run only from an upper-case `C:\` cwd (`cd "C:/Users/BS23-DESKTOP-00038/Projects/BanglaVoiceBotFineTuned/frontend"`).
- Baseline before the move: pytest 208 passed / 10 skipped; vitest 26 files / 166 tests.

## Review Focus

1. A real backend venv installs `bangla-voice-backend` with no extras: importing `app.core.contract` must not need FastAPI or SQLAlchemy (Task 2 guard test).
2. Test package names must not shadow the real `workers` package: `backend/tests/workers/` has an `__init__.py`, so `backend/tests/__init__.py` must exist to make its modules `tests.workers.*` (Task 1).
3. `backend_python()` must find the workspace root at `backend/` and recognise `workers/backends/<name>` in `members`/`exclude` (Task 1 tests).
4. Compose bind-mounts and `BACKENDS_ROOT` must point at `backend/workers/backends` on the host and inside the container at the same absolute path (Task 4).
5. Stale references to old paths or module names outside `docs/superpowers` (Task 5 grep).

---

### Task 1: Move the Python tree and rename imports

**Files:**
- Move: `pyproject.toml`, `uv.lock`, `.python-version`, `conftest.py` → `backend/`
- Move: `api/mlops_api` → `backend/app`; `shared/mlops_shared` → `backend/app/core`; `worker/mlops_worker` → `backend/workers`; `backends/fake` → `backend/workers/backends/fake`
- Move: `api/alembic`, `api/alembic.ini` → `backend/`; `api/Dockerfile` → `backend/Dockerfile` (contents fixed in Task 4)
- Move: `api/tests` → `backend/tests/app`; `shared/tests` → `backend/tests/core`; `worker/tests` → `backend/tests/workers`
- Create: `backend/tests/__init__.py` (empty)
- Delete: `api/pyproject.toml`, `shared/pyproject.toml`, `worker/pyproject.toml`
- Modify: `backend/pyproject.toml`, `backend/workers/backends/fake/pyproject.toml`, `backend/app/settings.py`, `backend/app/backends.py`, `backend/conftest.py`, `backend/tests/app/test_backends.py`, `backend/tests/app/test_settings.py`, `backend/tests/core/test_repo_catalog.py`, `backend/tests/app/test_app.py`, every `.py` under `backend/` (import rewrite)

**Interfaces:**
- Produces: packages `app`, `app.core`, `workers`; `app.app:create_app`; `python -m workers`; `Settings.backends_root` default `REPO_ROOT / "backend" / "workers" / "backends"`; uv project `bangla-voice-backend` with extras `validate`, `server`, `worker`.

- [ ] **Step 1: Close anything holding files open** (dev servers, `npm run dev`, terminals inside `api/`, `web/`). Windows refuses to rename a directory with open handles.

- [ ] **Step 2: Move files with `git mv`** in the order listed under Files. Move `shared/mlops_shared` into `backend/app/core` after `backend/app` exists. Remove the emptied `api/`, `shared/`, `worker/`, `backends/` directories (check each holds only `__pycache__`/`.venv` first). Delete the root `.venv` only after the new `backend/.venv` syncs in Step 6.

- [ ] **Step 3: Rewrite imports** with a scratchpad Python script over every `*.py` under `backend/`: regex `\bmlops_shared\b` → `app.core`, then `\bmlops_api\b` → `app`, then `\bmlops_worker\b` → `workers`. Files are UTF-8; preserve line endings. Then `git grep -nE "mlops_(api|shared|worker)" backend` must print nothing.

- [ ] **Step 4: Write `backend/pyproject.toml`**

```toml
[project]
name = "bangla-voice-backend"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = ["pydantic>=2.7", "pyyaml"]

[project.optional-dependencies]
validate = ["jsonschema>=4"]
server = [
  "jsonschema>=4", "fastapi>=0.110", "uvicorn>=0.29", "sqlalchemy[asyncio]>=2.0", "asyncpg", "aiosqlite",
  "alembic>=1.13", "pydantic-settings>=2.2", "pyarrow", "numpy", "soundfile", "pyloudnorm",
  "nvidia-ml-py", "sse-starlette", "python-multipart",
]
worker = ["bangla-voice-backend[server]", "mlflow-skinny>=2.14"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["app", "workers"]
exclude = ["workers/backends"]

[tool.uv]
package = true

[tool.uv.workspace]
members = ["workers/backends/fake"]
# Real backends live in workers/backends/<name>, have their own venv, and must be listed here.
exclude = []

[tool.uv.sources]
mlops-backend-fake = { workspace = true }

[dependency-groups]
dev = ["pytest", "pytest-asyncio", "httpx", "jsonschema", "mlops-backend-fake"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
testpaths = ["tests", "workers/backends/fake/tests"]
```

`backend/workers/backends/fake/pyproject.toml`: dependency `bangla-voice-backend`, source `bangla-voice-backend = { workspace = true }`.

- [ ] **Step 5: Fix path code**
  - `app/settings.py`: keep `REPO_ROOT = Path(__file__).resolve().parents[2]`; default `backends_root = REPO_ROOT / "backend" / "workers" / "backends"`.
  - `app/backends.py`: workspace root is `settings.backends_root.parents[1]`; member path is `f"workers/backends/{name}"`; root venv is `<workspace root>/.venv`. `BackendMissing` message unchanged.
  - `backend/conftest.py`: `backends_root=REPO / "workers" / "backends"` (`REPO` is `backend/`).
  - `tests/app/test_settings.py`, `tests/core/test_repo_catalog.py`: repo root is `Path(__file__).resolve().parents[3]`.
  - `tests/app/test_app.py`: alembic.ini is `Path(__file__).parents[2] / "alembic.ini"`.

- [ ] **Step 6: Update `tests/app/test_backends.py` to the new layout first, run it, watch it fail**

`_repo()` builds `tmp/repo/backend/workers/backends/fake`, writes `backend/pyproject.toml`, root venv at `backend/.venv`, and returns `backends_root=repo/"backend"/"workers"/"backends"`. Member strings become `"workers/backends/fake"`, `"workers/backends/*"`, exclude `"workers/backends/fake"`; the root-venv assertion becomes `s.backends_root.parents[1] / ".venv" / VENV_SUB`.

Run: `cd backend && uv sync --all-extras && uv run pytest tests/app/test_backends.py -q`
Expected before the `backends.py` fix: FAIL in `test_backend_python_workspace_member_uses_root_venv` and the excluded-backend test (old code looks at `backends_root.parent`). After the fix: PASS.

- [ ] **Step 7: Full suite**

Run: `cd backend && uv lock && uv sync --all-extras && uv run pytest -q`
Expected: 208 passed, 10 skipped.

- [ ] **Step 8: Commit** every moved/changed path explicitly (use `git status --porcelain` to list them), message `refactor: move Python code to backend/ with app, app.core and workers packages`.

---

### Task 2: Guard the light core install

**Files:**
- Test: `backend/tests/core/test_import_isolation.py`

**Interfaces:**
- Consumes: Task 1 packages.

- [ ] **Step 1: Write the test**

```python
import subprocess, sys

def test_core_contract_imports_without_server_deps():
    code = "import sys, app.core.contract, app.core.recipes, app.core.catalog; print(sorted(m for m in ('fastapi','sqlalchemy','pyarrow','mlflow') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]"
```

- [ ] **Step 2: Prove it can fail**: temporarily add `import fastapi` to `backend/app/__init__.py`, run `cd backend && uv run pytest tests/core/test_import_isolation.py -q`. Expected: FAIL with `"['fastapi']" == "[]"`. Remove the line.

- [ ] **Step 3: Run it.** Expected: PASS. If `app.core.recipes` or `app.core.catalog` pulls in a server dependency (e.g. jsonschema at import time is fine; FastAPI is not), make that import lazy.

- [ ] **Step 4: Base-install check**: `uv venv C:/<scratchpad>/corevenv --python 3.12` then `uv pip install --python <that venv> ./backend` and run `python -c "import app.core.contract"` with it. Expected: no error, and `pip list` shows no fastapi. Delete the scratch venv.

- [ ] **Step 5: Commit** `test: guard app.core against server dependencies`.

---

### Task 3: Move the web app to frontend/

**Files:**
- Move: `web` → `frontend` (`git mv web frontend`; untracked `node_modules`/`.next` move with the directory)
- Modify: `.gitignore` (`!web/src/**` → `!frontend/src/**`)

- [ ] **Step 1: `git mv web frontend`**, edit `.gitignore`.
- [ ] **Step 2: Verify** from `C:/Users/BS23-DESKTOP-00038/Projects/BanglaVoiceBotFineTuned/frontend`: `npx vitest run` → 26 files, 166 passed; `npx tsc --noEmit`, `npm run lint`, `npm run build` → clean. If `.next` caches the old path, delete `frontend/.next` and rebuild.
- [ ] **Step 3: Commit** `refactor: move web app to frontend/`.

---

### Task 4: Docker and compose

**Files:**
- Modify: `backend/Dockerfile`, `docker-compose.yml`
- Move: `.dockerignore` → `backend/.dockerignore`

- [ ] **Step 1: `backend/Dockerfile`** (build context `./backend`): keep base image, libsndfile, user `mlops` uid 1000. Copy `pyproject.toml uv.lock .python-version`, `app`, `workers`, `alembic`, `alembic.ini`. Run `uv sync --frozen --no-dev --extra server`. CMD `["uvicorn", "app.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]`. `backend/.dockerignore`: `.venv`, `**/.venv`, `**/__pycache__`, `tests`.
- [ ] **Step 2: `docker-compose.yml`**: api `build: {context: ./backend}`; command `alembic -c alembic.ini upgrade head && uvicorn app.app:create_app --factory ...`; `BACKENDS_ROOT: ${REPO_ROOT:?...}/backend/workers/backends`; volume `./backend/workers/backends:${REPO_ROOT}/backend/workers/backends:ro`; web `context: ./frontend`. Nothing else changes.
- [ ] **Step 3: Verify**: `POSTGRES_PASSWORD=x DATA_ROOT=/tmp/d REPO_ROOT=/r docker compose config -q` → exit 0 (works without the daemon). If `docker info` succeeds: `docker build ./backend` and `docker build ./frontend` → success. If the daemon is down, ledger a note that image builds are left to the VM smoke check.
- [ ] **Step 4: Commit** `build: point Docker and compose at backend/ and frontend/`.

---

### Task 5: Scripts, local runner and docs

**Files:**
- Modify: `scripts/setup_venvs.sh`, `scripts/dev.sh`, `run-local.bat`, `README.md`, `.env.example`

- [ ] **Step 1: `scripts/setup_venvs.sh`**: `uv sync --project backend --all-extras`; loop over `backend/workers/backends/*/pyproject.toml`, skip `fake`, check `"workers/backends/${name}"` is in `exclude` of `backend/pyproject.toml`; error text names that file and path. `uv run --project backend` for the workspace python line.
- [ ] **Step 2: `scripts/dev.sh`**: `exec uv run --project backend python -m workers`.
- [ ] **Step 3: `run-local.bat`**: `BACKENDS_ROOT=%ROOT%\backend\workers\backends`; `uv sync --frozen --all-extras --project backend`; `uv run --project backend alembic -c backend\alembic.ini upgrade head`; API `uv run --project backend uvicorn app.app:create_app ...`; worker `uv run --project backend python -m workers`; web from `%ROOT%\frontend` (both the `npm ci` check and the `start /d`).
- [ ] **Step 4: README and `.env.example`**: replace module names and paths (`mlops_worker` → `workers`, `shared/mlops_shared/catalog.py` → `backend/app/core/catalog.py`, `backends/<name>` → `backend/workers/backends/<name>`, test commands `cd backend && uv run pytest`, `cd frontend && npm test`, exclude entry `"workers/backends/<name>"` in `backend/pyproject.toml`). Add a short "Layout" section listing `backend/app`, `backend/app/core`, `backend/workers`, `backend/workers/backends`, `frontend`, `catalog`, `recipes`, `deploy`, `scripts`.
- [ ] **Step 5: Stale-reference sweep**: `git grep -nE "mlops_(api|shared|worker)|mlops-(api|shared|worker)|\b(api|shared|worker|web)/|(^|[^/])backends/" -- ':!docs/superpowers' ':!docs/context' ':!*.lock' ':!frontend/package-lock.json'`. Expected: only hits that are URL paths (`/api/...`) or `workers/backends/...`. Fix any other.
- [ ] **Step 6: Bash syntax check**: `bash -n scripts/setup_venvs.sh scripts/dev.sh` → no output.
- [ ] **Step 7: Final run**: `cd backend && uv run pytest -q` → 209 passed, 10 skipped (208 + the Task 2 guard); frontend vitest 166 passed.
- [ ] **Step 8: Commit** `docs: update scripts, local runner and README for the new layout`.
