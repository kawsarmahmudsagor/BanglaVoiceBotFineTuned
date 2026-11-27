# Bangla voice MLOps platform

A web UI and API for preparing Bangla speech data and running training and evaluation jobs on a GPU VM. Postgres, MLflow, the API and the web UI run in Docker Compose. The worker runs natively on the VM so training backends get the GPU and their own uv environments.

## Setup on the VM

Target: Ubuntu 24.04 with an NVIDIA driver, Docker with Compose, nvidia-container-toolkit, and uv.

1. Clone the repo and copy the config: `cp .env.example .env`. Set `DATA_ROOT` (a directory with plenty of disk), `REPO_ROOT` (absolute path of this checkout), and `POSTGRES_PASSWORD`. Put the same password in `DATABASE_URL`. Keep `.env` plain: no quotes, no spaces around `=`, and no `$` or `#` in values (the scripts also strip Windows line endings). The password goes into database URLs, so it must be URL-safe (letters and digits).
2. Create `DATA_ROOT` and make it writable by the VM user (`run-vm.sh` and `dev.sh` run the API and MLflow containers as that user's uid and gid). `dev.sh` creates it if missing and stops with a message if it is not writable.
3. Run `scripts/check_vm.sh`. It checks the GPU driver, the Docker nvidia runtime, Compose, uv and free disk, and prints `OK` or each failure.
4. Run `scripts/setup_venvs.sh` to sync the backend venv (`backend/.venv`) and one venv per real backend.
5. Run `scripts/dev.sh`. It starts the compose stack, waits for `/api/health`, and runs the worker in the foreground (`uv run --project backend --extra worker python -m workers`). Stop the worker with Ctrl-C; `docker compose down` stops the rest.

**One command:** `./run-vm.sh` (the Linux counterpart of `run-local.bat`) does steps 3 to 5: it checks `.env` (required keys, matching passwords, `REPO_ROOT` pointing at this checkout), runs `check_vm.sh` and `setup_venvs.sh`, creates the `mlflow` database if the Postgres volume predates it, starts the compose stack, waits for the API, and starts the worker in a detached tmux session (`mlops-worker`) with its output also in `$DATA_ROOT/logs/worker.log`. `./run-vm.sh --no-sync` skips the venv sync, `./run-vm.sh status` shows services and worker state, `./run-vm.sh logs` attaches to the worker, and `./run-vm.sh stop` stops the worker gracefully and then the compose stack.

Run `scripts/dev.sh` inside `tmux` (or as a systemd user unit) so the worker survives a dropped SSH connection: `tmux new -s mlops`, run the script, detach with Ctrl-b d, reattach with `tmux attach -t mlops`. If the worker does die mid-run, its next start stops the orphaned backend process group and marks the run failed.

`deploy/postgres/init.sql` (which creates the `mlflow` database) only runs on a fresh Postgres volume. On an existing volume, create it by hand: `docker compose exec postgres psql -U mlops -d mlops -c 'CREATE DATABASE mlflow;'`.

MLflow runs as your VM user (`UID`/`GID`, default 1000). Services listen on the VM's loopback only: web on 3000, API on 8000, MLflow on 5000, Postgres on 5432 (set `POSTGRES_PORT` in `.env`, and the same port in `DATABASE_URL`, if 5432 is taken).

## Reaching it from your PC

Open an SSH tunnel and browse `http://localhost:3000`:

```
ssh -L 3000:localhost:3000 -L 8000:localhost:8000 -L 5000:localhost:5000 <user>@<vm-ip>
```

The web UI calls the API at `http://localhost:8000` from your browser, so the 8000 forward is required. `WEB_ORIGIN` in `.env` must match the origin you browse from.

## VM smoke check

Run this once after setup to confirm the whole platform works end to end. It uses the `fake` backend, so it needs no model downloads and takes a few minutes.

1. On the VM, run `cp .env.example .env` (skip if done) and set `DATA_ROOT`, `REPO_ROOT`, `POSTGRES_PASSWORD` and the matching password in `DATABASE_URL`.
2. Run `scripts/check_vm.sh`. Success: the last line is `OK`.
3. Run `scripts/setup_venvs.sh`. Success: it exits 0 without errors.
4. Start `tmux new -s mlops` and run `scripts/dev.sh` inside it. Success: it prints the tunnel hint, then the worker starts and stays in the foreground. Detach with Ctrl-b d; the worker keeps running if SSH drops.
5. On your PC, open the tunnel from "Reaching it from your PC" and browse `http://localhost:3000`. Success: the dashboard loads.
6. Datasets > Upload: choose task `llm-sft` and upload a small Bangla JSONL file (sample below). Fill in source, license and consent. Success: the report shows no errors and a dataset version 1 appears.
7. Recipes > New: create a recipe named `smoke` with backend `fake`, task `llm-sft`, train dataset `<your dataset>@v1`, and hyperparameters `steps: 20` and `step_sleep_s: 0.5`. Success: validation passes and the recipe saves as v1.
8. Start a run from that recipe. Success: the run page opens, the phase timeline advances, and loss charts and log lines update live.
9. While it is training, click Cancel. Success: the run ends in `cancelled`.
10. Click Resume. Success: the run goes back to `queued`, then trains again.
11. Let it finish. Success: the state is `done` and the run page shows eval metrics and an export.
12. Click Promote to registry. Success: the page navigates to the new model's page under Registry.
13. Registry > open the model > Set as champion. Success: the Champion label appears on the model page and at the top of its lane in Registry.
14. Overview, active run: while a run is training (repeat step 8 if needed), the GPU card shows your GPU and the GPU lock card shows the run id as holder.
15. Overview, idle: with no run active, the GPU lock card shows the lock as free.
16. Open `http://localhost:5000` (MLflow, through the tunnel). Success: the run appears with its metrics.

Sample `smoke.jsonl` (each line is one chat; these 5 lines are enough for the smoke check, since the fake backend does not read the data and small val/test splits are fine):

```
{"messages": [{"role": "user", "content": "আজকের আবহাওয়া কেমন?"}, {"role": "assistant", "content": "আজ আকাশ মেঘলা, বিকেলে বৃষ্টি হতে পারে।"}]}
{"messages": [{"role": "user", "content": "ঢাকা থেকে চট্টগ্রাম কত দূর?"}, {"role": "assistant", "content": "সড়কপথে প্রায় ২৫০ কিলোমিটার।"}]}
{"messages": [{"role": "user", "content": "আমার বিলের অবস্থা জানতে চাই।"}, {"role": "assistant", "content": "আপনার অ্যাকাউন্ট নম্বরটি বলুন, আমি দেখে দিচ্ছি।"}]}
{"messages": [{"role": "user", "content": "অফিস কখন খোলে?"}, {"role": "assistant", "content": "সকাল ৯টা থেকে বিকেল ৫টা পর্যন্ত।"}]}
{"messages": [{"role": "user", "content": "ধন্যবাদ।"}, {"role": "assistant", "content": "আপনাকেও ধন্যবাদ, আর কিছু জানার থাকলে বলুন।"}]}
```

Assistant replies must be plain text: no markdown, lists or emoji.

If something fails:

- Worker problems (a run stuck in `queued`, a backend crash): read the worker terminal where `scripts/dev.sh` is running.
- API or UI errors: run `docker compose logs api` (and `docker compose logs web`, `docker compose logs mlflow`).
- A single run: open `$DATA_ROOT/runs/<run-id>/logs/` for that run's stdout and stderr, or the log panel on the run page.
- API not healthy: check `curl http://localhost:8000/api/health` on the VM, and that `.env` matches the rules in Setup.

## Where data lives

Everything is under `DATA_ROOT`:

- `data/incoming`, `data/datasets`, `data/prepared`: uploaded and prepared audio data.
- `runs/`: one directory per training or evaluation run, with logs and checkpoints.
- `mlflow/`: MLflow artifacts. Run and experiment metadata is in the `mlflow` Postgres database.

Platform records (datasets, jobs, runs) are in the `mlops` Postgres database, kept in the `pgdata` Docker volume. The API container sees `DATA_ROOT` and the repo's `backend/workers/backends/` and `recipes/` at the same absolute paths as the host, so paths it hands to the worker are valid on both sides.

## Model catalog

The Catalog page lists base models you can fine-tune, in three sections: LLM, STT and TTS. Each entry says what the model is, its license and Bangla support, the training data it needs (in the platform's upload format), and what to target when fine-tuning. "Fine-tune" opens a new recipe pre-filled from the entry.

- Entries are YAML files at `catalog/<section>/<id>.yaml`, where `<section>` is `llm`, `stt` or `tts` and `<id>` matches the file's `id` field. The schema is `CatalogEntry` in `backend/app/core/catalog.py`; unknown fields are rejected.
- To add a model, copy an existing entry in the same section, edit it, cite every claim in `sources`, and set `verified` to the date you checked the model card. No code change is needed.
- `cd backend && uv run pytest tests/core/test_repo_catalog.py` validates every committed entry. A file that fails to load also shows as a warning on the Catalog page, and the other entries still load.
- The API reads the directory in `CATALOG_DIR` (default `<repo>/catalog`; compose mounts it read-only).

## Training backends

Real training runs in six backends under `backend/workers/backends/`, each with its own uv environment:

| Backend | Tasks and models |
|---|---|
| `unsloth` | LLM SFT (Qwen3, Gemma 3, TituLM Llama 3.2; lora, qlora, full) and Whisper LoRA |
| `hf` | Whisper full or adapter, wav2vec2 and MMS CTC, Indic Parler-TTS |
| `nemo` | ASR (IndicConformer, BanglaConformer, Bengali FastConformer hybrid) |
| `f5` | TTS (IndicF5) |
| `cosyvoice` | TTS (CosyVoice 3, experimental for Bangla) |
| `qwen-asr` | ASR (Qwen3-ASR 0.6B Bengali; transformers 5, full fine-tune) |

Setup on the VM:

- System packages: `build-essential cmake` (the unsloth backend builds llama.cpp for GGUF export) and `ffmpeg` (decodes m4a/aac sources).
- `bash scripts/setup_venvs.sh [backend]` syncs the backend venvs (all of them without an argument). Lockfiles target Linux x86_64 only.
- Set `HF_TOKEN` in `.env` and accept the model terms on Hugging Face for the gated repos: `ai4bharat/IndicF5`, `ai4bharat/indic-parler-tts`, `ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large` and `google/gemma-3-4b-it`. Without access a run fails in prepare with `no access to <repo>: accept its terms on Hugging Face and set HF_TOKEN`. Weights are cached in `$DATA_ROOT/models/hf`.
- `bash scripts/check_backend.sh <name>` syncs one backend and checks torch, CUDA and its key imports.
- `bash scripts/test_backend.sh <name>` runs a backend's light tests (no GPU, no weights). Add `--heavy` for the GPU tests, on the VM only. Shared code: `cd backend && uv run pytest`.

### End-to-end smoke

With the stack and worker up (`scripts/dev.sh`), `bash scripts/smoke_backends.sh [backend...]` runs every backend (or the ones named) through the API. For each it:

1. ingests a 16-sample fixture (`scripts/smoke/fixtures/llm.jsonl`, or sine-tone `asr.zip`/`tts.zip` of 1.5 s clips that `scripts/smoke/make_audio_fixtures.py` builds at run time with `uv run` in `backend/`);
2. saves a recipe for the backend's first model with `hparams.max_steps=20` and `hparams.eval.max_samples=4`, and trains it;
3. checks the run is `done`, `/prepared` is non-empty, `eval.metrics` has `perplexity` (LLM), `wer` (ASR) or `rtf` (TTS), and every export artifact exists on disk;
4. starts a second run with `init_from: {run: <first run>, checkpoint: final}` and repeats the checks plus `parent_run_id`.

It prints `PASS <backend>/<model>` and `PASS <backend>/<model> warm-start`, or `FAIL ...` with the reason, and exits non-zero on any failure. Environment: `API` (default `http://localhost:8000/api`), `SKIP_BACKENDS` (space-separated; a backend skipped at its VM gate), `RUN_TIMEOUT` (seconds per run, default 3600). The script reads export paths from the host filesystem, so run it on the VM.

### Dataset format to prepared format

Upload formats are validated at ingest (`file,text` for ASR; `file,text,speaker` for TTS; chat JSONL for LLM). At run start each backend converts the dataset version into a prepared cache under `$DATA_ROOT/data/prepared/<dataset_hash>/<spec_hash>/`, shared by runs with the same conversion spec. Audio converters keep references to the uploaded files instead of copying audio, to stay within the disk budget; the exception is `qwen-asr`, which writes 16 kHz mono WAV copies (spec §17.3).

| Family | Backend | Input fields | Prepared output |
|---|---|---|---|
| `chat` | unsloth | `messages`, `tools` | Tokenized rows (JSONL) with the model's chat template applied and labels only on assistant turns; samples over `max_seq_len` are skipped |
| `whisper` | unsloth, hf | `file`, `text` | No audio: `audio_path`, duration and label ids (Bengali transcribe tokens). Audio is decoded to 16 kHz log-mel in the dataloader. Clips over 30 s or 448 label tokens are skipped |
| `ctc` | hf | `file`, `text` | No audio: `<split>.jsonl` of `audio_path`, duration, text, plus `vocab.json` (character vocabulary for wav2vec2; the `ben` adapter vocabulary for MMS) |
| `parler` | hf | `file`, `text`, `speaker` | DAC codes at 44.1 kHz (not audio), prompt and description token ids |
| `nemo-manifest` | nemo | `file`, `text` | `<split>.jsonl` manifests (`audio_filepath`, `duration`, `text`) pointing at the source files; only formats soundfile cannot read are transcoded to 16 kHz FLAC |
| `f5` | f5 | `file`, `text`, `speaker` | `<split>/raw.arrow` (audio path, text, duration), `duration.json`, `vocab.txt` (base vocabulary plus new characters appended) and one reference clip per speaker |
| `cosyvoice` | cosyvoice | `file`, `text`, `speaker` | Kaldi lists, speaker embeddings, speech tokens, and parquet shards that embed the source audio bytes; one reference clip per speaker |
| `qwen-asr` | qwen-asr | `file`, `text` | 16 kHz mono WAV copies of the clips plus `<split>.jsonl` (`audio` relative to the cache, `text`, `duration`); clips over 30 s or transcripts over `max_label_tokens` (448) tokens are skipped. The cache key includes the base model, so switching between the two base models re-prepares the WAV copies |

Text goes through `bn_normalize` first. Rows that cannot be converted are skipped and listed on the run page; a run fails if more than 5% are skipped (`hparams.prepare.max_skip_share`).

## Adding a backend

A backend is a directory under `backend/workers/backends/<name>/` that the worker runs as a subprocess in its own venv.

1. Add `capabilities.json` describing what the backend supports (tasks, parameters, outputs).
2. Add a Python package exposing the `mlops_backend` CLI that implements those capabilities. Use `backend/workers/backends/fake` as the reference.
3. Give it its own `pyproject.toml`, depending on `bangla-voice-backend` by path (`bangla-voice-backend = { path = "../../..", editable = true }` under `[tool.uv.sources]`) with no extras, so the backend gets `app.core.contract` without the API's server dependencies. It must not be a workspace member, so add `"workers/backends/<name>"` to `exclude` under `[tool.uv.workspace]` in `backend/pyproject.toml`.
4. Run `scripts/setup_venvs.sh`. It runs `uv sync --project backend/workers/backends/<name>` for every backend except `fake` and fails with a clear message if a backend is missing from `exclude`.

## The fake backend

`backend/workers/backends/fake` is a lightweight backend that simulates preparation, training and evaluation without a GPU or model downloads. It is a workspace member and is used by the test suite and for exercising the API, worker and UI end to end. It is also the reference implementation of the backend contract.

## Development

Python tests: `cd backend && uv sync --all-extras && uv run pytest`. Web tests: `cd frontend && npm ci && npm test`.

### Layout

- `backend/app`: the FastAPI service (package `app`). `backend/app/core` (package `app.core`) holds the recipe, catalog and backend-contract models; it needs only pydantic and PyYAML.
- `backend/workers`: the worker (`python -m workers`). `backend/workers/backends/<name>` holds the training backends.
- `backend/alembic`: database migrations. `backend/tests`: Python tests.
- `frontend`: the Next.js web UI.
- `catalog`, `recipes`: model catalog entries and saved recipes. `deploy`: compose support files. `scripts`: VM scripts.

`backend/pyproject.toml` is the uv workspace root. Its extras: `server` (API dependencies), `worker` (`server` plus MLflow). The Docker image installs `server`; the VM venv and `run-local.bat` install all extras.

### Running everything locally on Windows

Double-click `run-local.bat` (or run it from a terminal). It installs dependencies, migrates the database, opens three windows (API on :8000, worker, web on :3000) and opens the browser.

- If Docker Desktop is running, Postgres and MLflow (http://localhost:5000) start from `deploy/local/docker-compose.yml`. Otherwise it falls back to SQLite with MLflow disabled. `run-local.bat sqlite` forces SQLite.
- Data goes to `.\var` (git-ignored). The script's settings override the VM values in `.env`.
- `run-local.bat stop` closes the three windows and stops the Docker services.
- Requires `uv` and Node.js 20+. Training uses the fake backend locally; real backends run on the VM.
