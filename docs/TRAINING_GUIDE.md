# Run the platform and fine-tune Whisper large-v3-turbo (Bengali)

This guide takes you from a fresh checkout to a finished fine-tuning run on the GPU machine. It uses the Bengali.AI
subset (about 200 h) and the `mozilla-ai/whisper-large-v3-turbo-bn` base model with the `hf` backend (full fine-tune).

Steps marked **(once)** only need doing the first time.

## 0. What you need

- Ubuntu with an NVIDIA driver (the dev machine has an RTX 5090, 32 GB).
- Docker with Compose and `nvidia-container-toolkit`, `uv`, `tmux`, `curl`, `unzip`, `ffmpeg`, `build-essential`, `cmake`, `git`.
- A Hugging Face token in `.env` (`HF_TOKEN`). The Whisper models are not gated, but keep the token set.
- Disk: the subset zip is 4.1 GB, the unzipped import copy about 4 GB, and the ingested dataset version about 4 GB more.
  Checkpoints of a full fine-tune are several GB each. Keep at least 100 GB free under `DATA_ROOT`.

## 1. Build the dataset zip (once)

The raw data is the Kaggle `bengaliai-speech` competition, already downloaded to
`~/.cache/kagglehub/competitions/bengaliai-speech` (see `dataset_kaggle.py`).

```bash
python3 scripts/make_bengaliai_subset.py --out /home/xr23/bengaliai_200h.zip
```

Result: 160,000 train clips (about 200 h) and 3,000 validation clips in `metadata_train.csv` and `metadata_valid.csv`.
For a different size, use `--train 80000` (about 100 h) or `--train 240000` (about 300 h). Train and validation share
no sentences, and no sentence appears in more than two train clips.

## 2. Configure `.env` (once)

```bash
cp .env.example .env        # skip if .env already exists
```

Set these keys. Format rules: `KEY=value`, no quotes, no spaces around `=`, no `$` or `#` in values.

| Key | Value |
|---|---|
| `DATA_ROOT` | `/srv/mlops` (must exist and be writable by your user) |
| `REPO_ROOT` | `/home/xr23/Projects/BanglaVoiceBotFineTuned` (must be this checkout) |
| `POSTGRES_PASSWORD` | letters and digits only, not `change-me` |
| `DATABASE_URL` | `postgresql+asyncpg://mlops:<same password>@localhost:5432/mlops` |
| `HF_TOKEN` | your Hugging Face token |
| `DEFAULT_DATASET_ZIP` | `/home/xr23/bengaliai_200h.zip` (this is the default) |

If port 5432 is already taken (the machine also runs other Postgres containers), set `POSTGRES_PORT=5433` and use
`5433` in `DATABASE_URL` as well. `run-vm.sh` checks the ports and tells you which one is busy.

Make sure `DATA_ROOT` exists and is yours:

```bash
sudo mkdir -p /srv/mlops && sudo chown "$(id -u):$(id -g)" /srv/mlops
```

## 3. Start everything

```bash
./run-vm.sh
```

What it does, in order:

1. Checks `.env` and runs `scripts/check_vm.sh` (GPU driver, Docker nvidia runtime, uv, compilers, disk).
2. Syncs the uv environments (`scripts/setup_venvs.sh`). The first run is slow because it installs every backend.
   Later restarts can use `./run-vm.sh --no-sync`.
3. Starts Postgres, MLflow, the API and the web UI in Docker, and waits for `http://localhost:8000/api/health`.
4. Starts the worker in a detached tmux session `mlops-worker`. Its log is `$DATA_ROOT/logs/worker.log`.
5. Registers the default dataset `bengaliai-200h` (`scripts/seed_default_dataset.sh`): it unzips the zip into
   `$DATA_ROOT/import/bengaliai-200h` and queues an ingest. The worker validates the 163,000 clips and creates
   dataset version 1.

Useful commands:

```bash
./run-vm.sh status    # services, API health, worker state
./run-vm.sh logs      # attach to the worker (detach: Ctrl-b d)
./run-vm.sh stop      # stop the worker gracefully, then the Docker services
```

If `check_vm.sh` fails, it prints every problem. Fix them and run `./run-vm.sh` again.

### Dev mode: no Docker image builds

Use this while you are fixing code. Docker mode rebuilds the API and web images on every start, which is slow.
Dev mode runs everything from the source tree, so edits apply without any build.

```bash
./run-vm.sh dev
```

What it does, in order:

1. Stops the Docker-mode MLflow, API and web containers if they run, because they use the same ports.
2. Starts only Postgres in Docker. It uses the stock `postgres:16` image, so nothing is built.
3. Syncs the backend venv and runs the database migrations natively.
4. Runs `npm ci` in `frontend/` the first time only.
5. Starts three windows in the tmux session `mlops-dev`:
   - `mlflow`: the MLflow server on port 5000, from a cached `uvx` environment. The first start downloads MLflow,
     so it takes a few minutes.
   - `api`: uvicorn on port 8000 with `--reload`.
   - `web`: `next dev` on `WEB_PORT`.
6. Starts the worker in the tmux session `mlops-worker` and seeds the default dataset, as in Docker mode.

The output of each service is also written to `$DATA_ROOT/logs/mlflow.log`, `api.log` and `web.log`.

What to do after an edit:

| You changed | What to do |
|---|---|
| API code (`backend/app/`) | Nothing. The API restarts by itself. |
| Web code (`frontend/`) | Nothing. The page reloads by itself. |
| Worker or training code (`backend/workers/`, `backend/app/core/`) | `./run-vm.sh restart-worker`. This also stops a running training job, so do it between runs. |
| `.env` | `./run-vm.sh stop`, then `./run-vm.sh dev`. |
| Python or npm dependencies | `./run-vm.sh stop`, then `./run-vm.sh dev`. For a training backend's venv, also run `bash scripts/setup_venvs.sh <backend>`. |

Useful commands:

```bash
tmux attach -t mlops-dev        # see MLflow, API and web; switch windows with Ctrl-b n, detach with Ctrl-b d
./run-vm.sh restart-worker      # restart only the worker
./run-vm.sh status              # shows the dev session too
./run-vm.sh stop                # stops the worker, the dev session and Postgres
```

To go back to Docker mode, run `./run-vm.sh stop`, then `./run-vm.sh`.

## 4. Open the UI

The web UI listens on `WEB_PORT` from `.env` (default 3000).
On this VM, port 3000 is used by another project, so `.env` sets `WEB_PORT=3001` and `WEB_ORIGIN=http://localhost:3001`.

If you work on the GPU machine itself, browse `http://localhost:3001`.
From another PC, open a tunnel first and then browse `http://localhost:3001`:

```bash
ssh -L 3001:localhost:3001 -L 8000:localhost:8000 -L 5000:localhost:5000 xr23@10.112.145.130
```

The 8000 forward is required because the browser calls the API directly. MLflow is at `http://localhost:5000`.
`./run-vm.sh` prints the exact tunnel command for the current `.env` when it finishes.

## 5. Wait for the dataset to be ready

Open **Datasets**. `bengaliai-200h` appears once ingest finishes, with version 1 and about 163,000 samples. Ingest of
this size takes a while. Progress is on the dataset page, and errors are in `$DATA_ROOT/logs/worker.log`.

Check the validation report on the version page. If it lists errors, the dataset cannot be used until they are fixed.
You can delete the unzipped copy in `$DATA_ROOT/import/bengaliai-200h` after ingest succeeds to save about 4 GB.

If the dataset never showed up, run the seeding step by hand while the API is up:

```bash
bash scripts/seed_default_dataset.sh
```

## 6. Create the training recipe

1. Go to **Catalog > STT > Whisper large-v3-turbo Bengali (mozilla-ai)** and click **Fine-tune**.
   This opens `/train/new?catalog=whisper-large-v3-turbo-bn` pre-filled.
2. Check the form:
   - Task `asr`, backend `hf`, method `full`.
   - Base model `mozilla-ai/whisper-large-v3-turbo-bn`.
   - Train dataset: already set to `bengaliai-200h@v1`. You do not need to pick it.
3. Set the hyperparameters in the YAML panel. These are the values to start from for 200 h on a 32 GB GPU:

```yaml
hparams:
  learning_rate: 1.0e-5
  epochs: 2
  batch_size: 8
  grad_accum: 4
  warmup_steps: 500
  gradient_checkpointing: true
  keep_checkpoints: 2
  eval:
    baseline: true
    max_samples: 2000
    max_val_samples: 1000
export: [ct2]
```

   - Always set `learning_rate` yourself. If it is missing, the backend falls back to `2e-4`, which is far too high for a
     full fine-tune of Whisper and will damage the model.
   - Effective batch is `batch_size x grad_accum` = 32. With 160,000 clips that is 5,000 steps per epoch, so 2 epochs is
     about 10,000 steps. To cap the run instead, set `max_steps` (it overrides `epochs`).
   - The catalog default of `max_steps: 5000` is too short for 200 h: it would cover only one epoch.
   - If you hit a CUDA out-of-memory error, lower `batch_size` to 4 and raise `grad_accum` to 8.
4. Give the recipe a name, for example `whisper-bn-200h`, and click **Validate**, then **Save**.

## 7. Start the run

1. Open the recipe under **Train**, pick version 1, and click **Start run**.
2. The run page opens. The phases go in order: prepare, train, eval, export.
   - **prepare**: converts the dataset into label ids and a prepared cache under `$DATA_ROOT/data/prepared/`. Audio is not
     copied. A run fails if more than 5% of rows are skipped (clips over 30 s or 448 label tokens are skipped).
   - **train**: loss charts and log lines update live. The worker saves a checkpoint and evaluates every
     `max(50, total_steps / 10)` steps.
   - **eval**: computes WER and CER on the validation split, plus the same metrics for the base model
     (`eval.baseline: true`) so you can see the gain.
   - **export**: writes the CTranslate2 model (`ct2`) for `faster_whisper` / RealtimeSTT.
3. Watch GPU use with `nvidia-smi`, or the GPU card on the Overview page. Only one run holds the GPU at a time.

## 8. Stop, resume, cancel

- **Cancel** on the run page stops the run gracefully and saves a checkpoint.
- **Resume** continues from the last checkpoint.
- If the machine reboots, run `./run-vm.sh`. The worker marks the orphaned run as failed on start. Start a new run from the same recipe, or use **Resume** if the run page offers it.

## 9. After training

1. Check the metrics on the run page: WER and CER against the baseline. Both should drop clearly. Compare the same
   normalisation each time, because orthographic WER is much higher than normalised WER.
2. If WER is still falling at the end, train longer or on more data. If it stopped improving early, you have enough
   data for this setup.
3. Click **Promote to registry**, then on the model page **Set as champion** if this is your best model.
4. Find the files under `$DATA_ROOT/runs/<run-id>/`: checkpoints, logs and the exported model. MLflow
   (`http://localhost:5000`) holds the metrics.
5. To scale up, rerun step 1 with `--train 240000 --out /home/xr23/bengaliai_300h.zip`, set `DEFAULT_DATASET_NAME` and
   `DEFAULT_DATASET_ZIP` to match, and change the dataset name in `frontend/src/components/recipe-editor.tsx`
   (`DEFAULT_DATASET`), or upload the new zip in the UI under **Datasets > New** and select it in the recipe.

## 10. Troubleshooting

| Symptom | What to check |
|---|---|
| `run-vm.sh` stops with a port error | Another service uses 5432, 5000, 8000 or 3000. Free it, or set `POSTGRES_PORT` and the matching `DATABASE_URL`. |
| `DATA_ROOT ... not writable` | Run the `sudo mkdir -p` / `chown` command from step 2. |
| Run stays `queued` | The worker is down. `./run-vm.sh status`, then `./run-vm.sh logs`. |
| Dataset missing after startup | The zip path is wrong or the worker has not finished. Check `$DATA_ROOT/logs/worker.log`, then run `bash scripts/seed_default_dataset.sh`. |
| API or web errors | `docker compose logs api` or `docker compose logs web`. |
| One run's output | `$DATA_ROOT/runs/<run-id>/logs/` or the log panel on the run page. |
| CUDA out of memory | Lower `batch_size`, raise `grad_accum`, keep `gradient_checkpointing: true`. |
| Loss is `nan` | The run stops itself. Lower `learning_rate` (for example `5e-6`) and start a new run. |
| Many skipped rows in prepare | Look at the skipped list on the run page. Fix the transcripts or clip lengths, then re-ingest. |

## 11. Quick check that the platform works (optional, no GPU needed)

Run the `fake` backend end to end before spending hours on a real run. Follow "VM smoke check" in `README.md`.
