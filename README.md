# Bangla voice MLOps platform

A web UI and API for preparing Bangla speech data and running training and evaluation jobs on a GPU VM. Postgres, MLflow, the API and the web UI run in Docker Compose. The worker runs natively on the VM so training backends get the GPU and their own uv environments.

## Setup on the VM

Target: Ubuntu 24.04 with an NVIDIA driver, Docker with Compose, nvidia-container-toolkit, and uv.

1. Clone the repo and copy the config: `cp .env.example .env`. Set `DATA_ROOT` (a directory with plenty of disk), `REPO_ROOT` (absolute path of this checkout), and `POSTGRES_PASSWORD`. Put the same password in `DATABASE_URL`. The password goes into database URLs, so it must be URL-safe (letters and digits).
2. Create `DATA_ROOT` and make it writable by uid 1000 (the API container user, normally the first VM user). `dev.sh` creates it if missing and stops with a message if it is not writable.
3. Run `scripts/check_vm.sh`. It checks the GPU driver, the Docker nvidia runtime, Compose, uv and free disk, and prints `OK` or each failure.
4. Run `scripts/setup_venvs.sh` to sync the workspace venv and one venv per real backend.
5. Run `scripts/dev.sh`. It starts the compose stack, waits for `/api/health`, and runs the worker in the foreground (`uv run python -m mlops_worker`). Stop the worker with Ctrl-C; `docker compose down` stops the rest.

Services listen on the VM's loopback only: web on 3000, API on 8000, MLflow on 5000, Postgres on 5432.

## Reaching it from your PC

Open an SSH tunnel and browse `http://localhost:3000`:

```
ssh -L 3000:localhost:3000 -L 8000:localhost:8000 -L 5000:localhost:5000 <user>@<vm-ip>
```

The web UI calls the API at `http://localhost:8000` from your browser, so the 8000 forward is required. `WEB_ORIGIN` in `.env` must match the origin you browse from.

## Where data lives

Everything is under `DATA_ROOT`:

- `data/incoming`, `data/datasets`, `data/prepared`: uploaded and prepared audio data.
- `runs/`: one directory per training or evaluation run, with logs and checkpoints.
- `mlflow/`: MLflow artifacts. Run and experiment metadata is in the `mlflow` Postgres database.

Platform records (datasets, jobs, runs) are in the `mlops` Postgres database, kept in the `pgdata` Docker volume. The API container sees `DATA_ROOT` and the repo's `backends/` and `recipes/` at the same absolute paths as the host, so paths it hands to the worker are valid on both sides.

## Adding a backend

A backend is a directory under `backends/<name>/` that the worker runs as a subprocess in its own venv.

1. Add `capabilities.json` describing what the backend supports (tasks, parameters, outputs).
2. Add a Python package exposing the `mlops_backend` CLI that implements those capabilities. Use `backends/fake` as the reference.
3. Give it its own `pyproject.toml`. It must not be a workspace member, so add `"backends/<name>"` to `exclude` under `[tool.uv.workspace]` in the root `pyproject.toml`.
4. Run `scripts/setup_venvs.sh`. It runs `uv sync --project backends/<name>` for every backend except `fake` and fails with a clear message if a backend is missing from `exclude`.

## The fake backend

`backends/fake` is a lightweight backend that simulates preparation, training and evaluation without a GPU or model downloads. It is a workspace member and is used by the test suite and for exercising the API, worker and UI end to end. It is also the reference implementation of the backend contract.

## Development

`uv sync` then `uv run pytest` runs the tests for all workspace packages.
