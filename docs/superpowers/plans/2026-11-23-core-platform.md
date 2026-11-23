# Core Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build sub-project 1 of the Bangla Voice MLOps platform: upload and version datasets, write recipes, run model-agnostic training jobs through the run-directory contract (proved with a fake backend), promote runs to a model registry, and do all of it from a Next.js UI.

**Architecture:** A uv workspace holds three Python packages: `mlops_shared` (run-dir contract and recipe schema, stdlib + pydantic only, installable into every backend venv), `mlops_api` (FastAPI, async SQLAlchemy, dataset ingest) and `mlops_worker` (database-backed job loop, GPU lock, subprocess runner). Backends are separate projects under `backends/`, each exposing `python -m mlops_backend <phase> --run-dir`; this plan ships only `backends/fake`. Postgres, MLflow, API and web run in docker-compose; the worker runs natively on the VM.

**Tech Stack:** Python 3.12, uv workspace, FastAPI, SQLAlchemy 2 (async; asyncpg in prod, aiosqlite in tests), Alembic, pydantic v2 + pydantic-settings, jsonschema, PyYAML, pyarrow, soundfile, numpy, pyloudnorm, nvidia-ml-py, mlflow-skinny, sse-starlette, pytest + pytest-asyncio + httpx. Web: Next.js 15 App Router, React 19, TypeScript, Tailwind, shadcn/ui, tus-js-client, recharts, @uiw/react-codemirror + @codemirror/lang-yaml, yaml, vitest + @testing-library/react.

**Spec:** `docs/superpowers/specs/2026-11-23-bangla-voice-mlops-design.md` (sections §3, §4, §5, §9, §10, §11, §12, §13 item 1). Integration context: `docs/context/localvoiceagent.md`.

## Global Constraints

- Python `>=3.12,<3.13`. The VM has Python 3.12.3 and `uv`.
- Tests run with no GPU, network, Docker or model weights: `uv run pytest` from the repo root, `npm run test && npm run typecheck && npm run build` from `web/`.
- `mlops_shared` depends only on the stdlib and `pydantic>=2.7`. It must never import torch, numpy or the API.
- The API and worker never import backend code. They only write `config.json`, start `python -m mlops_backend <phase> --run-dir <dir>` and read files.
- Phases in order: `prepare`, `train`, `eval`, `export`. Run states: `queued`, `preparing`, `training`, `evaluating`, `exporting`, `done`, `failed`, `cancelled`.
- Task names exactly: `llm-sft`, `asr`, `tts`.
- One GPU lock. A run holds it from claim to final state. Training is never preempted.
- Cancel: SIGTERM, then SIGKILL after `CANCEL_GRACE_S` (default 60).
- Disk: warn below `DISK_WARN_GB` (default 20) free on `DATA_ROOT`; refuse new uploads and runs below `DISK_MIN_GB` (default 10) with HTTP 507.
- Default split ratios 90/5/5 (train/val/test), assigned by hash of sample ID.
- Audio warnings: duration outside 0.5–30 s, silence share above 50 %, loudness outside −30 to −14 LUFS, clipping, mixed sample rates.
- `commercial_ok` of a run = base model flag AND every training dataset version's flag.
- No auth. CORS allows only `WEB_ORIGIN` (default `http://localhost:3000`).
- Shell scripts use LF line endings (`.gitattributes`). Stage files by explicit path; never `git add -A`. Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Deviations from spec, approved with this plan: the job queue is database-backed (no Redis); the `llm-sft` "too long" warning uses total characters > 12,000 per sample instead of a tokenizer.

## Review Focus

1. **Zip from Windows Explorer**: the archive has one top-level folder and Bangla or space-containing file names. Ingest must find `metadata.csv` inside that single folder and read every file. Test in Task 6.
2. **`metadata.csv` saved by Excel**: UTF-8 with BOM and CRLF line endings. Must parse with correct column names. Test in Task 6.
3. **Upload resumed after an API restart**: HEAD must report the offset from the bytes on disk, not from memory, and PATCH must continue from there. Test in Task 5.
4. **Backend that ignores SIGTERM**: cancel must still end the run as `cancelled` after the grace period via SIGKILL and release the GPU lock. Test in Task 10.
5. **Promote clicked twice**: the second call returns the same model version (200), not a duplicate. Test in Task 13.

---

## File Structure

```
pyproject.toml                    uv workspace root, pytest config
conftest.py                       shared fixtures (tmp DATA_ROOT, sqlite DB, ASGI client, fake backend env)
.gitattributes  .env.example  docker-compose.yml
shared/pyproject.toml
shared/mlops_shared/contract.py   RunDir + config/status/metrics/eval/export models and IO
shared/mlops_shared/recipes.py    RecipeSpec, Capabilities, validate_recipe, parse_dataset_ref, commercial_ok
shared/tests/
backends/fake/pyproject.toml  backends/fake/capabilities.json
backends/fake/mlops_backend/__main__.py   fake phases
backends/fake/tests/
api/pyproject.toml  api/alembic.ini  api/alembic/  api/Dockerfile
api/mlops_api/settings.py         Settings (pydantic-settings), disk helpers
api/mlops_api/db.py               engine/session factory, Base
api/mlops_api/models.py           ORM tables
api/mlops_api/app.py              create_app(), router wiring, CORS
api/mlops_api/backends.py         discover capabilities, backend python path
api/mlops_api/uploads.py          tus router
api/mlops_api/datasets/validate_llm.py  validate_audio.py  report.py  ingest.py  versions.py
api/mlops_api/routers/datasets.py  recipes.py  runs.py  models.py  system.py
api/mlops_api/system.py           read_gpu, disk_status
api/tests/
worker/pyproject.toml
worker/mlops_worker/lock.py       GPU lock
worker/mlops_worker/jobs.py       claim, recovery
worker/mlops_worker/executor.py   run phases as subprocesses
worker/mlops_worker/ingest_job.py CPU ingest job
worker/mlops_worker/tracking.py   MLflow forwarder
worker/mlops_worker/__main__.py   main loop
worker/tests/
scripts/setup_venvs.sh  scripts/check_vm.sh  scripts/dev.sh
web/                              Next.js app (Tasks 16–20)
recipes/                          read-only YAML copies of recipes
```

---

### Task 1: Workspace and run-directory contract

**Files:**
- Create: `pyproject.toml`, `.gitattributes`, `shared/pyproject.toml`, `shared/mlops_shared/__init__.py`, `shared/mlops_shared/contract.py`
- Test: `shared/tests/test_contract.py`

**Interfaces:**
- Produces (in `mlops_shared.contract`):
  - `class Phase(str, Enum)`: `prepare`, `train`, `eval`, `export`
  - `class DatasetRef(BaseModel)`: `name: str, version: int, hash: str, root: str, manifest: str, splits: list[str]`
  - `class RunConfig(BaseModel)`: `run_id: str, attempt: int, resume: bool, task: str, backend: str, base_model: str, method: dict, hparams: dict, export: list[str], seed: int, datasets: dict[str, DatasetRef], prepared_root: str`
  - `class Status(BaseModel)`: `phase: Phase, step: int = 0, total_steps: int | None = None, eta_s: float | None = None, message: str = "", updated_at: float`
  - `class Metric(BaseModel)`: `step: int, split: str, name: str, value: float, ts: float`
  - `class ExportArtifact(BaseModel)`: `format: str, path: str, size_bytes: int, serve_with: str | None = None`
  - `class ExportManifest(BaseModel)`: `artifacts: list[ExportArtifact]`
  - `class EvalResults(BaseModel)`: `metrics: dict[str, float], samples_dir: str | None = None`
  - `class RunDir`: `__init__(self, path: Path)`; properties `config_path, status_path, metrics_path, logs_dir, checkpoints_dir, eval_dir, export_dir`; `ensure() -> None`
  - `write_config(rd, cfg) / read_config(rd) -> RunConfig`
  - `write_status(rd, st) / read_status(rd) -> Status | None`
  - `append_metric(rd, m) / read_metrics(rd, offset: int = 0) -> tuple[list[Metric], int]`
  - `write_eval_results / read_eval_results(rd) -> EvalResults | None`
  - `write_export_manifest / read_export_manifest(rd) -> ExportManifest | None`

- [ ] **Step 1: Create the workspace.** Root `pyproject.toml`: `[tool.uv.workspace] members = ["shared", "api", "worker", "backends/fake"]`; `[tool.pytest.ini_options] asyncio_mode = "auto"`, `testpaths = ["shared/tests", "api/tests", "worker/tests", "backends/fake/tests"]`; dev group with `pytest`, `pytest-asyncio`, `httpx`. `.gitattributes`: `*.sh text eol=lf`, `* text=auto`. `shared/pyproject.toml`: package `mlops-shared`, dependency `pydantic>=2.7`. (`api`, `worker`, `backends/fake` pyprojects are added in their tasks; until then list only `shared` in members.)

- [ ] **Step 2: Write the failing tests**

```python
def test_status_roundtrip_is_atomic(tmp_path):
    rd = RunDir(tmp_path / "r1"); rd.ensure()
    write_status(rd, Status(phase=Phase.train, step=3, total_steps=10, updated_at=1.0))
    assert read_status(rd).step == 3
    assert not list(rd.path.glob("*.tmp"))

def test_read_status_missing_or_partial_returns_none(tmp_path):
    rd = RunDir(tmp_path / "r1"); rd.ensure()
    assert read_status(rd) is None
    rd.status_path.write_text('{"phase": "tr')
    assert read_status(rd) is None

def test_read_metrics_ignores_trailing_partial_line(tmp_path):
    rd = RunDir(tmp_path / "r1"); rd.ensure()
    append_metric(rd, Metric(step=1, split="train", name="loss", value=0.5, ts=1.0))
    with rd.metrics_path.open("a") as f:
        f.write('{"step": 2, "spl')
    ms, offset = read_metrics(rd)
    assert [m.step for m in ms] == [1]
    with rd.metrics_path.open("a") as f:
        f.write('it": "train", "name": "loss", "value": 0.4, "ts": 2.0}\n')
    ms2, _ = read_metrics(rd, offset)
    assert [m.step for m in ms2] == [2]

def test_config_roundtrip(tmp_path): ...  # write_config then read_config equals original
```

- [ ] **Step 3: Run to verify failure.** `uv run pytest shared/tests/test_contract.py -v` → FAIL (`ModuleNotFoundError: mlops_shared`).

- [ ] **Step 4: Implement `contract.py`.** Writes go to `<file>.tmp` then `os.replace`. `read_status` returns `None` on missing file or `ValidationError`/`JSONDecodeError`. `read_metrics` reads bytes from `offset`, parses only complete lines (ending in `\n`), returns the byte offset after the last complete line. Directory names: `logs`, `checkpoints`, `eval`, `export`; files `config.json`, `status.json`, `metrics.jsonl`, `eval/results.json`, `export/manifest.json`.

- [ ] **Step 5: Run to verify pass.** `uv run pytest shared/tests -v` → PASS.

- [ ] **Step 6: Commit** `feat(shared): run-directory contract`

---

### Task 2: Recipe schema and validation

**Files:**
- Create: `shared/mlops_shared/recipes.py`
- Test: `shared/tests/test_recipes.py`

**Interfaces:**
- Produces (in `mlops_shared.recipes`):
  - `TASKS = ("llm-sft", "asr", "tts")`
  - `class BaseModelMeta(BaseModel)`: `license: str, commercial_ok: bool`
  - `class RecipeDatasets(BaseModel)`: `train: str, eval: str | None = None` (refs `name@vN`)
  - `class RecipeSpec(BaseModel)`: `name: str` (kebab-case slug), `task: Literal[...]`, `backend: str`, `base_model: str`, `base_model_meta: BaseModelMeta`, `datasets: RecipeDatasets`, `method: dict` (must contain `type`), `hparams: dict`, `export: list[str] = []`; property `seed -> int` = `hparams.get("seed", 42)`
  - `class TaskCapability(BaseModel)`: `methods: list[str], export_formats: list[str], serve_formats: list[str], schema_: dict = Field(alias="schema")`
  - `class Capabilities(BaseModel)`: `name: str, label: str, tasks: dict[str, TaskCapability]`
  - `parse_dataset_ref(ref: str) -> tuple[str, int]` raises `ValueError`
  - `validate_recipe(spec: RecipeSpec, caps: dict[str, Capabilities]) -> list[str]`
  - `commercial_ok(base: BaseModelMeta, dataset_flags: Iterable[bool]) -> bool`
- `jsonschema` is imported lazily inside `validate_recipe` and added as an optional extra `mlops-shared[validate]`; API depends on that extra. Backends do not need it.

- [ ] **Step 1: Write the failing tests**

```python
def test_parse_dataset_ref():
    assert parse_dataset_ref("ecom-instruct@v3") == ("ecom-instruct", 3)
    for bad in ["ecom", "ecom@3", "ecom@v0", "Ecom@v1"]:
        with pytest.raises(ValueError): parse_dataset_ref(bad)

def test_validate_recipe_reports_each_problem():
    caps = {"fake": Capabilities(name="fake", label="Fake", tasks={"llm-sft": TaskCapability(
        methods=["none"], export_formats=["dummy"], serve_formats=[],
        schema={"type": "object", "properties": {"hparams": {"type": "object",
                "properties": {"steps": {"type": "integer", "minimum": 1}}}}})})}
    spec = make_spec(task="asr", backend="fake", method={"type": "lora"},
                     hparams={"steps": 0}, export=["gguf:q4_k_m"])
    errs = validate_recipe(spec, caps)
    assert any("task 'asr' not supported by backend 'fake'" in e for e in errs)
    errs = validate_recipe(make_spec(task="llm-sft", backend="fake", method={"type": "lora"},
                                     hparams={"steps": 0}, export=["gguf:q4_k_m"]), caps)
    assert "method 'lora' not supported" in " ".join(errs)
    assert "export format 'gguf' not supported" in " ".join(errs)
    assert any(e.startswith("hparams.steps") for e in errs)
    assert validate_recipe(make_spec(backend="nope"), caps) == ["unknown backend 'nope'"]

def test_commercial_ok():
    yes = BaseModelMeta(license="apache-2.0", commercial_ok=True)
    assert commercial_ok(yes, [True, True])
    assert not commercial_ok(yes, [True, False])
    assert not commercial_ok(BaseModelMeta(license="cc-by-nc", commercial_ok=False), [True])

def test_recipe_name_must_be_slug(): ...      # "Qwen 4B" raises ValidationError
def test_method_requires_type(): ...         # method={} raises ValidationError
```
`make_spec(**over)` is a test helper returning a valid `RecipeSpec` with overrides.

- [ ] **Step 2: Run to verify failure.** `uv run pytest shared/tests/test_recipes.py -v` → FAIL.

- [ ] **Step 3: Implement `recipes.py`.** Slug regex `^[a-z0-9]+(-[a-z0-9]+)*$`; ref regex `^([a-z0-9]+(?:-[a-z0-9]+)*)@v([1-9][0-9]*)$`. `validate_recipe` checks in order: backend known (return immediately if not), task supported (return immediately if not), method type, each export format (text before `:`), then JSON Schema over `{"method": ..., "hparams": ...}` with `Draft202012Validator`, each error formatted `"<dotted.path>: <message>"`.

- [ ] **Step 4: Run to verify pass.** `uv run pytest shared/tests -v` → PASS.

- [ ] **Step 5: Commit** `feat(shared): recipe schema and validation`

---

### Task 3: Fake backend

**Files:**
- Create: `backends/fake/pyproject.toml`, `backends/fake/capabilities.json`, `backends/fake/mlops_backend/__init__.py`, `backends/fake/mlops_backend/__main__.py`
- Test: `backends/fake/tests/test_fake_backend.py`
- Modify: root `pyproject.toml` (add `backends/fake` to members)

**Interfaces:**
- Consumes: `mlops_shared.contract` (Task 1).
- Produces: CLI `python -m mlops_backend {prepare|train|eval|export|serve} --run-dir DIR`. Behavior from `config.hparams`: `steps: int = 5`, `step_sleep_s: float = 0.0`, `fail_phase: str | None`, `ignore_sigterm: bool = False`, `oom: bool = False`. `capabilities.json`: `name "fake"`, `label "Fake (pipeline test)"`, all three tasks, `methods ["none"]`, `export_formats ["dummy"]`, `serve_formats []`, schema with `hparams.steps` integer ≥ 1.

- [ ] **Step 1: Write the failing tests** (each runs the CLI with `subprocess.run([sys.executable, "-m", "mlops_backend", phase, "--run-dir", str(dir)])` on a `RunConfig` written by a helper)

```python
def test_full_pipeline_writes_contract_files(run_dir):
    for phase in ["prepare", "train", "eval", "export"]:
        assert run(phase, run_dir).returncode == 0
    rd = RunDir(run_dir)
    ms, _ = read_metrics(rd)
    assert [m.step for m in ms if m.name == "loss"] == [1, 2, 3, 4, 5]
    assert read_eval_results(rd).metrics == {"fake_score": 0.5}
    art = read_export_manifest(rd).artifacts[0]
    assert art.format == "dummy" and (rd.export_dir / art.path).exists()

def test_fail_phase_exits_nonzero(run_dir_with(fail_phase="train")): ...   # returncode == 1
def test_oom_prints_cuda_oom_to_stderr(...): ...                         # "CUDA out of memory" in stderr, rc 1
def test_sigterm_saves_checkpoint_and_resume_continues(run_dir_with(steps=50, step_sleep_s=0.05)):
    p = popen("train"); wait_until(lambda: (rd.checkpoints_dir).glob("step-*")); p.send_signal(SIGTERM)
    assert p.wait(5) == 143
    last = max_checkpoint_step(rd)
    set_resume(rd, True); set_steps(rd, last + 2)
    assert run("train", run_dir).returncode == 0
    assert min(m.step for m in new_metrics) == last + 1
```
Skip the SIGTERM test on Windows (`pytest.mark.skipif(sys.platform == "win32")`).

- [ ] **Step 2: Run to verify failure.** `uv sync && uv run pytest backends/fake/tests -v` → FAIL.

- [ ] **Step 3: Implement `__main__.py`.** Each phase writes `Status(phase=...)` first. `train` writes a checkpoint file `checkpoints/step-<n>.txt` and a `loss = 1/(n+1)` metric each step; on resume starts after the highest checkpoint step. SIGTERM handler (unless `ignore_sigterm`) writes a checkpoint and `sys.exit(143)`. `serve` prints `"fake backend cannot serve"` and exits 2.

- [ ] **Step 4: Run to verify pass.** → PASS.

- [ ] **Step 5: Commit** `feat(backends): fake backend implementing the run-dir contract`

---

### Task 4: API skeleton, settings, database, migrations

**Files:**
- Create: `api/pyproject.toml`, `api/mlops_api/{__init__,settings,db,models,app,backends}.py`, `api/alembic.ini`, `api/alembic/env.py`, `api/alembic/versions/0001_initial.py`, root `conftest.py`
- Modify: root `pyproject.toml` (add `api` to members)
- Test: `api/tests/test_app.py`, `api/tests/test_backends.py`

**Interfaces:**
- Produces:
  - `Settings` (pydantic-settings, env): `database_url: str`, `data_root: Path`, `backends_root: Path`, `recipes_dir: Path`, `web_origin: str = "http://localhost:3000"`, `mlflow_tracking_uri: str | None = None`, `disk_warn_gb: float = 20`, `disk_min_gb: float = 10`, `cancel_grace_s: float = 60`, `poll_interval_s: float = 2`; properties `incoming_dir`, `datasets_dir`, `prepared_dir`, `runs_dir` under `data_root/data/...` and `data_root/runs`. `get_settings()` cached.
  - `db.py`: `Base`, `make_engine(url)`, `make_sessionmaker(engine)`, FastAPI dependency `get_session()`.
  - `models.py` tables (all `id` primary keys, `created_at` timestamps, JSON via `sqlalchemy.JSON`):
    - `Upload(id: str uuid, filename, length: int, state: "uploading"|"complete")`
    - `Dataset(id, name unique, task, eval_only: bool)`
    - `DatasetVersion(id, dataset_id FK, number: int, hash: str, path: str, n_samples: int, report: JSON, provenance: JSON, split_ratios: JSON, parent_version_id: int | None)`; unique `(dataset_id, number)`
    - `Ingest(id, upload_id FK, dataset_name, task, eval_only, provenance: JSON, state: "pending"|"running"|"done"|"failed", report: JSON | None, version_id: int | None, error: str | None)`
    - `Recipe(id, name unique)`, `RecipeVersion(id, recipe_id FK, number, spec: JSON, yaml: str, frozen: bool)`
    - `Run(id: str "r-<8 hex>", recipe_version_id FK, state, attempt: int = 1, resume: bool = False, cancel_requested: bool = False, run_dir: str, commercial_ok: bool, error: str | None, oom: bool = False, git_sha: str | None, lock_hash: str | None, dataset_hashes: JSON, mlflow_run_id: str | None, started_at, finished_at)`
    - `ModelVersion(id, name, number, run_id FK unique, task, base_model, metrics: JSON, exports: JSON, commercial_ok: bool, champion: bool = False)`
    - `GpuLock(id = 1, holder_kind: str | None, holder_id: str | None, acquired_at)`
  - `create_app(settings: Settings | None = None) -> FastAPI` with CORS for `web_origin`, `GET /api/health -> {"ok": true}`.
  - `backends.py`: `load_capabilities(backends_root) -> dict[str, Capabilities]` (reads `*/capabilities.json`, skips invalid with a logged warning); `backend_python(name, settings) -> Path` (env `BACKEND_PYTHON_<NAME>` upper-cased, else `backends_root/<name>/.venv/bin/python`; raises `BackendMissing("backend venv missing; run scripts/setup_venvs.sh")` if absent).
  - Root `conftest.py` fixtures: `settings` (tmp `data_root`, `backends_root` = repo `backends/`, sqlite `database_url` in tmp, `BACKEND_PYTHON_FAKE=sys.executable` via monkeypatch), `session_factory` (creates tables with `Base.metadata.create_all` and the `gpu_lock` row), `session` (one `AsyncSession` from it), `client` (`httpx.AsyncClient(transport=ASGITransport(app))`).

- [ ] **Step 1: Write the failing tests**

```python
async def test_health(client):
    assert (await client.get("/api/health")).json() == {"ok": True}

async def test_cors_allows_web_origin_only(client):
    r = await client.options("/api/health", headers={"Origin": "http://localhost:3000",
                                                     "Access-Control-Request-Method": "GET"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"
    r = await client.options("/api/health", headers={"Origin": "http://evil",
                                                     "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in r.headers

def test_load_capabilities_finds_fake(settings):
    assert load_capabilities(settings.backends_root)["fake"].tasks["llm-sft"].methods == ["none"]

def test_backend_python_missing_venv_raises(settings, monkeypatch):
    monkeypatch.delenv("BACKEND_PYTHON_FAKE")
    with pytest.raises(BackendMissing): backend_python("fake", settings)

def test_alembic_head_matches_models(tmp_path): ...  # run `alembic upgrade head` on a tmp sqlite, then compare table names with Base.metadata
```

- [ ] **Step 2: Run to verify failure.** → FAIL.

- [ ] **Step 3: Implement** the files above. Write the migration by autogenerate against sqlite, then review it by hand. `alembic/env.py` reads `DATABASE_URL` from `Settings`.

- [ ] **Step 4: Run to verify pass.** `uv run pytest api/tests -v` → PASS.

- [ ] **Step 5: Commit** `feat(api): app skeleton, settings, schema and migration`

---

### Task 5: Resumable uploads (tus core) and disk guard

**Files:**
- Create: `api/mlops_api/uploads.py`, `api/mlops_api/system.py` (disk part only)
- Modify: `api/mlops_api/app.py` (include router)
- Test: `api/tests/test_uploads.py`

**Interfaces:**
- Produces:
  - `disk_status(settings) -> DiskStatus(free_gb: float, warn: bool, block: bool)` (`shutil.disk_usage(data_root)`)
  - tus 1.0.0 core + creation extension at `/api/uploads`:
    - `OPTIONS /api/uploads` → 204, `Tus-Version: 1.0.0`, `Tus-Extension: creation`
    - `POST /api/uploads` with `Upload-Length`, `Upload-Metadata` (`filename <base64>`) → 201, `Location: /api/uploads/{id}`; 507 if `block` or `Upload-Length > (free_gb - disk_min_gb) GB`
    - `HEAD /api/uploads/{id}` → 200, `Upload-Offset` = current file size on disk, `Upload-Length`, `Cache-Control: no-store`
    - `PATCH /api/uploads/{id}` with `Content-Type: application/offset+octet-stream`, `Upload-Offset` → 204 with new offset; 409 on offset mismatch; 415 on wrong content type; sets `state="complete"` when offset reaches length
  - All responses carry `Tus-Resumable: 1.0.0` and expose tus headers via CORS (`expose_headers=["Location","Upload-Offset","Upload-Length","Tus-Resumable"]`).
  - Files live at `incoming_dir/<id>/<sanitized filename>`; sanitize to the basename.

- [ ] **Step 1: Write the failing tests**

```python
async def test_upload_in_two_chunks_completes(client, session_factory):
    loc = (await create(client, length=10, filename="ডেটা.jsonl")).headers["location"]
    assert (await patch(client, loc, offset=0, body=b"01234")).headers["upload-offset"] == "5"
    assert (await patch(client, loc, offset=5, body=b"56789")).status_code == 204
    assert (await upload_row(session_factory, loc)).state == "complete"

async def test_wrong_offset_is_409(client): ...

async def test_head_offset_survives_api_restart(settings, session_factory):   # Review Focus 3
    c1 = client_for(create_app(settings)); loc = (await create(c1, length=10)).headers["location"]
    await patch(c1, loc, offset=0, body=b"0123")
    c2 = client_for(create_app(settings))
    assert (await c2.head(loc)).headers["upload-offset"] == "4"
    assert (await patch(c2, loc, offset=4, body=b"456789")).status_code == 204

async def test_create_refused_when_disk_low(client, monkeypatch):
    monkeypatch.setattr("mlops_api.system.shutil.disk_usage", lambda p: usage(free_gb=5))
    assert (await create(client, length=10)).status_code == 507
```

- [ ] **Step 2: Run to verify failure.** → FAIL.
- [ ] **Step 3: Implement** `uploads.py` and `disk_status`. Append chunk bytes with `open(path, "ab")`; offset is always `path.stat().st_size`.
- [ ] **Step 4: Run to verify pass.** → PASS.
- [ ] **Step 5: Commit** `feat(api): resumable tus uploads with disk guard`

---

### Task 6: Dataset validators and ingest

**Files:**
- Create: `api/mlops_api/datasets/{__init__,report,validate_llm,validate_audio,ingest}.py`
- Test: `api/tests/datasets/test_validate_llm.py`, `test_validate_audio.py`, `test_ingest.py` (audio fixtures generated with numpy + soundfile in a fixture)

**Interfaces:**
- Produces:
  - `report.py`: `Issue(code: str, message: str, sample_id: str | None = None)`, `Report(errors: list[Issue], warnings: list[Issue], stats: dict)`; `Report.ok -> bool` (no errors)
  - `validate_llm(path: Path) -> tuple[Report, list[dict]]`. Rows: `{"sample_id", "messages_json", "tools_json" | None}`. Error codes `invalid_json`, `bad_role_order`, `empty_assistant`. Warning codes `low_bangla_share`, `markdown_in_assistant`, `emoji_in_assistant`, `duplicate`, `too_long`.
  - `validate_audio(root: Path, task: str) -> tuple[Report, list[dict]]`. Rows: `{"sample_id", "file", "text", "duration_s", "sample_rate", **optional columns}`. Required columns: `tts` → `file,text,speaker`; `asr` → `file,text`. Error codes `missing_metadata`, `missing_column`, `undecodable`, `missing_transcript`, `missing_audio`. Warning codes `duration_out_of_range`, `clipping`, `high_silence`, `loudness_out_of_range`, `mixed_sample_rates`.
  - `assign_split(sample_id: str, ratios: tuple[float, float, float] = (0.9, 0.05, 0.05)) -> str`
  - `ingest(upload_file: Path, task: str, dest_root: Path) -> IngestResult(report: Report, version_hash: str | None, version_dir: Path | None, n_samples: int)` — writes `dest_root/<hash16>/manifest.parquet` and `files/` only when `report.ok`.
- Rules (exact):
  - Role order regex over role letters `s,u,a,t`: `^s?(u(a(t+a)*))+$`.
  - Bangla share: letters in U+0980–U+09FF ÷ all `str.isalpha()` letters in assistant text; warn below 0.5.
  - Markdown: any assistant line matching `^\s*(#{1,6}\s|[-*]\s|\d+\.\s|\|)` or containing `**` or a backtick.
  - Emoji: any char in U+1F300–U+1FAFF or U+2600–U+27BF.
  - Duplicate: identical `messages_json` after `json.dumps(sort_keys=True, ensure_ascii=False)`.
  - Too long: total characters of all message contents > 12,000.
  - `sample_id` = first 16 hex of SHA-256 of the canonical row content (for audio: transcript + audio file SHA-256).
  - Clipping: fraction of samples with `abs >= 0.999` > 0.001. Silence: 20 ms frames with RMS below −50 dBFS > 50 %. Loudness: `pyloudnorm.Meter(sr).integrated_loudness` outside [−30, −14]; skip for clips < 0.4 s.
  - Zip root: `metadata.csv` at the archive root, or inside the archive's single top-level directory. Read CSV with `encoding="utf-8-sig"`, `newline=""`.
  - Version hash: SHA-256 over rows sorted by `sample_id`, each row serialized with `json.dumps(sort_keys=True, ensure_ascii=False)`. Rows include `split`. Directory name = first 16 hex.

- [ ] **Step 1: Write the failing tests**

```python
def test_llm_valid_sample_has_no_issues(tmp_path):
    p = jsonl(tmp_path, [{"messages": [{"role": "user", "content": "অর্ডার কোথায়?"},
                                       {"role": "assistant", "content": "চেক করে জানাচ্ছি।"}]}])
    report, rows = validate_llm(p)
    assert report.ok and not report.warnings and len(rows) == 1

@pytest.mark.parametrize("roles,ok", [("ua", True), ("sua", True), ("uata", True),
                                      ("au", False), ("uu", False), ("uat", False), ("", False)])
def test_llm_role_order(tmp_path, roles, ok): ...

def test_llm_warnings(tmp_path):
    report, _ = validate_llm(jsonl(tmp_path, [conv("hello there"), conv("- **দাম** ৳৫০০"),
                                              conv("ধন্যবাদ 😊"), conv("ধন্যবাদ 😊"), conv("ক" * 12001)]))
    assert codes(report.warnings) >= {"low_bangla_share", "markdown_in_assistant",
                                      "emoji_in_assistant", "duplicate", "too_long"}

def test_llm_invalid_json_line_is_error(tmp_path): ...

def test_audio_zip_with_top_folder_and_bangla_names(tmp_path):            # Review Focus 1
    z = make_zip(tmp_path, top="My Dataset", files={"audio/নমুনা ১.wav": sine(2.0)},
                 metadata="file,text\naudio/নমুনা ১.wav,আমার অর্ডার\n")
    result = ingest(z, "asr", tmp_path / "out")
    assert result.report.ok and result.n_samples == 1

def test_metadata_csv_with_bom_and_crlf(tmp_path):                         # Review Focus 2
    z = make_zip(tmp_path, files={"audio/a.wav": sine(2.0)},
                 metadata="﻿file,text\r\naudio/a.wav,হ্যালো\r\n")
    assert ingest(z, "asr", tmp_path / "out").report.ok

def test_audio_checks(tmp_path):
    report, _ = validate_audio(dir_with({"short.wav": sine(0.2), "loud.wav": clipped(2.0),
                                         "quiet.wav": silence(2.0), "sr.wav": sine(2.0, sr=8000)}), "asr")
    assert codes(report.warnings) >= {"duration_out_of_range", "clipping", "high_silence",
                                      "mixed_sample_rates"}

def test_audio_errors(tmp_path): ...   # missing metadata.csv, missing column 'speaker' for tts, row pointing at absent file, wav without row, corrupt wav

def test_assign_split_is_deterministic_and_roughly_90_5_5():
    splits = [assign_split(f"{i:016x}") for i in range(20000)]
    assert splits == [assign_split(f"{i:016x}") for i in range(20000)]
    assert 0.88 < splits.count("train") / 20000 < 0.92

def test_ingest_same_content_same_hash_and_errors_write_nothing(tmp_path): ...
```

- [ ] **Step 2: Run to verify failure.** → FAIL.
- [ ] **Step 3: Implement** the four modules. `ingest` extracts zips to a staging dir under `dest_root/.staging/<uuid>`, moves accepted audio into `<hash16>/files/` (keeping relative paths), and deletes staging in a `finally`.
- [ ] **Step 4: Run to verify pass.** → PASS.
- [ ] **Step 5: Commit** `feat(api): dataset validators and ingest`

---

### Task 7: Datasets API and transcript edits

**Files:**
- Create: `api/mlops_api/datasets/versions.py`, `api/mlops_api/routers/datasets.py`
- Modify: `api/mlops_api/app.py`
- Test: `api/tests/test_datasets_api.py`

**Interfaces:**
- Consumes: `Upload`, `Dataset`, `DatasetVersion`, `Ingest` (Task 4); `ingest` (Task 6). Ingest jobs are executed by the worker (Task 11); this task's tests call `run_ingest(session, ingest_id, settings)` directly.
- Produces:
  - `run_ingest(session, ingest_id: int, settings) -> None` in `versions.py`: sets `running`, calls `ingest` with `dest_root = settings.datasets_dir / <dataset name>`, on success creates `DatasetVersion` with `number = max + 1` (creating `Dataset` if new), sets `done` + `version_id`; on report errors sets `failed` + `report`; on exception sets `failed` + `error`.
  - `edit_transcript(session, version: DatasetVersion, sample_id: str, text: str, settings) -> DatasetVersion`: new version dir with the manifest row changed (and the sample re-hashed), audio files hardlinked (fallback copy), `parent_version_id` set, provenance copied.
  - Routes:
    - `POST /api/datasets/ingest` body `{upload_id, name, task, eval_only, provenance: {source, license, consent, commercial_ok, notes}}` → 202 `{ingest_id}`; 409 upload not complete; 409 `"dataset '<name>' is task '<task>'"` when name exists with another task; 422 bad slug/task.
    - `GET /api/ingests/{id}` → `{state, report, version, error}`
    - `GET /api/datasets` → `[{name, task, eval_only, latest: {number, n_samples, commercial_ok, created_at}}]`
    - `GET /api/datasets/{name}` → dataset + versions list
    - `GET /api/datasets/{name}/versions/{n}` → version incl. report, provenance, split counts
    - `GET /api/datasets/{name}/versions/{n}/samples?offset=0&limit=50&split=&warning=` → `{total, rows}`; `warning` filters by issue code via report `sample_id`s
    - `GET /api/datasets/{name}/versions/{n}/audio/{sample_id}` → `FileResponse` (404 for llm datasets)
    - `POST /api/datasets/{name}/versions/{n}/edits` `{sample_id, text}` → 201 new version summary (400 for `llm-sft`)
    - `PATCH /api/datasets/{name}` `{eval_only: bool}`

- [ ] **Step 1: Write the failing tests**

```python
async def test_ingest_flow_creates_v1_then_v2(client, session_factory, settings):
    up = await upload_bytes(client, "a.jsonl", jsonl_bytes(2))
    iid = (await client.post("/api/datasets/ingest", json=req(up, "ecom-instruct"))).json()["ingest_id"]
    await run_ingest_now(session_factory, iid, settings)
    assert (await client.get(f"/api/ingests/{iid}")).json()["state"] == "done"
    # second upload, same name -> v2
    ...
    assert [v["number"] for v in (await client.get("/api/datasets/ecom-instruct")).json()["versions"]] == [1, 2]

async def test_name_reused_with_other_task_is_409(client, ...): ...
async def test_failed_validation_keeps_report_and_creates_no_version(client, ...): ...
async def test_samples_filter_by_warning(client, ...): ...
async def test_edit_transcript_creates_child_version(client, ...):
    r = await client.post("/api/datasets/stt-a/versions/1/edits", json={"sample_id": sid, "text": "নতুন"})
    assert r.status_code == 201 and r.json()["number"] == 2
    rows = (await client.get("/api/datasets/stt-a/versions/2/samples")).json()["rows"]
    assert "নতুন" in [x["text"] for x in rows]
    assert (await client.get("/api/datasets/stt-a/versions/1/samples")).json()["rows"][0]["text"] != "নতুন"
```

- [ ] **Step 2: Run to verify failure.** → FAIL.
- [ ] **Step 3: Implement.** Read manifests with `pyarrow.parquet`.
- [ ] **Step 4: Run to verify pass.** → PASS.
- [ ] **Step 5: Commit** `feat(api): datasets API, ingest records and transcript edits`

---

### Task 8: Recipes API

**Files:**
- Create: `api/mlops_api/routers/recipes.py`
- Test: `api/tests/test_recipes_api.py`

**Interfaces:**
- Consumes: `RecipeSpec`, `validate_recipe`, `parse_dataset_ref` (Task 2); `load_capabilities` (Task 4); dataset tables.
- Produces:
  - `check_recipe(session, spec, caps) -> list[str]`: `validate_recipe` errors plus: each dataset ref exists (`"dataset 'x@v2' not found"`); dataset task equals recipe task; train dataset not `eval_only` (`"dataset 'x' is eval-only and cannot be used for training"`).
  - Routes:
    - `GET /api/backends` → list of `Capabilities` (by alias)
    - `POST /api/recipes/validate` body `{yaml}` or `{spec}` → `{errors: [...], spec}` (YAML parse errors reported as `"yaml: <msg>"`)
    - `POST /api/recipes` same body → 201 `{name, number}`; 422 `{errors}` if any. If the latest version is not frozen, it is overwritten; otherwise a new version is created. Writes `recipes_dir/<name>.yaml` with a header comment `# Generated from the platform database. Edit in the UI.`
    - `GET /api/recipes`, `GET /api/recipes/{name}` (versions), `GET /api/recipes/{name}/versions/{n}`
    - `POST /api/recipes/{name}/clone` `{new_name}` → 201

- [ ] **Step 1: Write the failing tests** — valid recipe saves v1 and writes the YAML file; save again while unfrozen keeps number 1; after `frozen=True` set in DB, save creates 2; eval-only train dataset error text; missing dataset error text; YAML syntax error; clone copies spec with new name.
- [ ] **Step 2: Run to verify failure.** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run to verify pass.** → PASS.
- [ ] **Step 5: Commit** `feat(api): recipes API with versioning and YAML export`

---

### Task 9: GPU lock, job claiming and recovery

**Files:**
- Create: `worker/pyproject.toml` (depends on `mlops-api`, `mlops-shared`, `mlflow-skinny`), `worker/mlops_worker/{__init__,lock,jobs}.py`
- Modify: root `pyproject.toml` (add `worker` to members)
- Test: `worker/tests/test_lock.py`, `worker/tests/test_jobs.py`

**Interfaces:**
- Produces:
  - `lock.py`: `async acquire(session, kind: str, holder_id: str) -> bool`, `async release(session, kind: str, holder_id: str) -> None` (no-op if not holder), `async current(session) -> GpuLock`. Implement as a conditional `UPDATE gpu_lock SET ... WHERE id = 1 AND holder_kind IS NULL`; check `rowcount == 1`. The row is created by the migration and by `ensure_lock_row(session)`.
  - `jobs.py`:
    - `async claim_next_run(session) -> Run | None`: oldest `queued` run; acquire the lock for it; conditional update `state='preparing' WHERE id=:id AND state='queued'`; on lost race release lock and return `None`. Returns `None` without touching anything if the lock is held.
    - `async claim_next_ingest(session) -> Ingest | None`: oldest `pending` → `running` (no GPU lock).
    - `async recover(session) -> list[str]`: runs in `preparing|training|evaluating|exporting` → `failed` with `error="worker restarted during <state>"`; ingests in `running` → `failed`; release any lock held by a run. Returns recovered run IDs.

- [ ] **Step 1: Write the failing tests**

```python
async def test_lock_is_exclusive(session):
    assert await acquire(session, "run", "r-1")
    assert not await acquire(session, "serve", "llm-1")
    await release(session, "serve", "llm-1")       # not holder: no-op
    assert (await current(session)).holder_id == "r-1"
    await release(session, "run", "r-1")
    assert await acquire(session, "serve", "llm-1")

async def test_claim_takes_oldest_and_holds_lock(session): ...
async def test_claim_returns_none_while_serve_holds_lock(session): ...
async def test_recover_marks_running_failed_and_frees_lock(session):
    run = await make_run(session, state="training"); await acquire(session, "run", run.id)
    assert await recover(session) == [run.id]
    assert (await get_run(session, run.id)).state == "failed"
    assert (await current(session)).holder_kind is None
```

- [ ] **Step 2: Run to verify failure.** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run to verify pass.** → PASS.
- [ ] **Step 5: Commit** `feat(worker): GPU lock, job claiming and restart recovery`

---

### Task 10: Run executor

**Files:**
- Create: `worker/mlops_worker/executor.py`
- Test: `worker/tests/test_executor.py`

**Interfaces:**
- Consumes: `RunDir`, `write_config`, `read_status` (Task 1); `backend_python` (Task 4); lock (Task 9); `Run`, `RecipeVersion`, `DatasetVersion`.
- Produces:
  - `build_config(run: Run, recipe: RecipeSpec, versions: dict[str, DatasetVersion], settings) -> RunConfig`: `datasets["train"]` has `splits=["train","val"]`; `datasets["eval"]` has `splits=["train","val","test"]` when the eval dataset is `eval_only`, else `["test"]`. `prepared_root = settings.prepared_dir`. `seed = recipe.seed`.
  - `PHASE_STATE = {"prepare": "preparing", "train": "training", "eval": "evaluating", "export": "exporting"}`
  - `async execute_run(session_factory, run_id: str, settings, forwarder: Forwarder) -> str` returns the final state. Steps: write `config.json`; record `git_sha` (`git rev-parse HEAD` at repo root, else `None`) and `lock_hash` (SHA-256 of `backends/<name>/uv.lock` if present); for each phase set state, spawn `backend_python(...) -m mlops_backend <phase> --run-dir <dir>` with stdout/stderr appended to `logs/stdout.log` / `logs/stderr.log`; while it runs, poll every 1 s: forward new metrics, and if `cancel_requested` send SIGTERM, wait `cancel_grace_s`, then SIGKILL; non-zero exit → `failed`, `error` = last non-empty stderr line, `oom = True` if logs match `CUDA out of memory|OutOfMemoryError`; all phases OK → `done`. Always sets `finished_at`, releases the lock, and calls `forwarder.finish(state)`.
  - `class Forwarder(Protocol)`: `start(run_id: str, params: dict, tags: dict) -> str | None`, `metrics(items: list[Metric]) -> None`, `finish(state: str) -> None`. `NullForwarder` implements it as no-ops (real one in Task 12).

- [ ] **Step 1: Write the failing tests** (use the fake backend via `BACKEND_PYTHON_FAKE=sys.executable`; `RecordingForwarder` collects calls)

```python
async def test_successful_run_reaches_done(env):
    run = await queued_run(env, hparams={"steps": 3})
    assert await execute_run(env.sf, run.id, env.settings, env.fwd) == "done"
    assert [m.step for m in env.fwd.metrics_seen] == [1, 2, 3]
    assert (await lock_state(env)).holder_kind is None

async def test_failed_phase_records_error_and_oom(env):
    run = await queued_run(env, hparams={"oom": True})
    assert await execute_run(...) == "failed"
    r = await get_run(env, run.id); assert r.oom and "CUDA out of memory" in r.error

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
async def test_cancel_sigkills_backend_that_ignores_sigterm(env):          # Review Focus 4
    env.settings.cancel_grace_s = 1
    run = await queued_run(env, hparams={"steps": 1000, "step_sleep_s": 0.05, "ignore_sigterm": True})
    task = asyncio.create_task(execute_run(env.sf, run.id, env.settings, env.fwd))
    await wait_for_state(env, run.id, "training"); await set_cancel(env, run.id)
    assert await asyncio.wait_for(task, 10) == "cancelled"
    assert (await lock_state(env)).holder_kind is None

async def test_build_config_eval_splits(env): ...   # eval_only -> all splits, otherwise ["test"]
async def test_missing_venv_fails_run_with_message(env, monkeypatch): ...   # error contains "setup_venvs.sh"
```

- [ ] **Step 2: Run to verify failure.** → FAIL.
- [ ] **Step 3: Implement** with `asyncio.create_subprocess_exec`. Use a fresh session per DB update so the cancel flag set by the API is seen.
- [ ] **Step 4: Run to verify pass.** → PASS.
- [ ] **Step 5: Commit** `feat(worker): run executor over the run-dir contract`

---

### Task 11: Worker main loop and ingest job

**Files:**
- Create: `worker/mlops_worker/ingest_job.py`, `worker/mlops_worker/__main__.py`
- Test: `worker/tests/test_main.py`

**Interfaces:**
- Consumes: `claim_next_run`, `claim_next_ingest`, `recover` (Task 9); `execute_run` (Task 10); `run_ingest` (Task 7).
- Produces:
  - `async gpu_tick(sf, settings, forwarder_factory) -> bool` (claims and executes one run; returns whether it did work)
  - `async cpu_tick(sf, settings) -> bool` (claims and runs one ingest in `asyncio.to_thread`)
  - `async main(settings) -> None`: `recover`, then two concurrent loops (`gpu_tick`, `cpu_tick`) sleeping `poll_interval_s` when idle; logs each claim. Entry `python -m mlops_worker`.

- [ ] **Step 1: Write the failing tests** — `cpu_tick` turns a pending ingest into `done`; `gpu_tick` returns `False` with no queued runs and `True` after executing one; an ingest runs while a run holds the GPU lock.
- [ ] **Step 2: Run to verify failure.** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run to verify pass.** → PASS.
- [ ] **Step 5: Commit** `feat(worker): main loop with GPU and CPU queues`

---

### Task 12: MLflow forwarding

**Files:**
- Create: `worker/mlops_worker/tracking.py`
- Test: `worker/tests/test_tracking.py`

**Interfaces:**
- Produces:
  - `flatten(d: dict, prefix: str = "") -> dict[str, str]` (`{"method": {"r": 16}}` → `{"method.r": "16"}`)
  - `class MlflowForwarder(Forwarder)`: `__init__(uri: str, client=None)`; `start` creates experiment `bangla-voice-<task>` if missing, a run named after the run ID, logs flattened params, sets tags `backend`, `base_model`, `commercial_ok`, `git_sha`; `metrics` logs with `step`, key `<split>/<name>`; `finish` maps `done→FINISHED`, `failed→FAILED`, `cancelled→KILLED`. Any MLflow exception is logged and swallowed (training never fails because tracking is down).
  - `make_forwarder(settings) -> Forwarder` (`NullForwarder` when `mlflow_tracking_uri` is unset).

- [ ] **Step 1: Write the failing tests** with a stub client object recording calls: `flatten`; key naming `train/loss`; status mapping; exception in `log_metric` does not raise.
- [ ] **Step 2: Run to verify failure.** → FAIL.
- [ ] **Step 3: Implement** with `mlflow.tracking.MlflowClient`.
- [ ] **Step 4: Run to verify pass.** → PASS.
- [ ] **Step 5: Commit** `feat(worker): forward run metrics to MLflow`

---

### Task 13: Runs, models and system API

**Files:**
- Create: `api/mlops_api/routers/runs.py`, `api/mlops_api/routers/models.py`, `api/mlops_api/routers/system.py`; extend `api/mlops_api/system.py` with `read_gpu`
- Test: `api/tests/test_runs_api.py`, `api/tests/test_models_api.py`, `api/tests/test_system_api.py`

**Interfaces:**
- Consumes: contract readers (Task 1); `check_recipe` (Task 8); `commercial_ok` (Task 2); `disk_status` (Task 5); `current` lock (Task 9).
- Produces:
  - Runs:
    - `POST /api/runs` `{recipe, version?}` (default latest) → 201 `{id}`; freezes the recipe version; re-runs `check_recipe` (422 on errors); 507 if disk `block`; computes `commercial_ok`, `dataset_hashes`; `run_dir = runs_dir/<id>`.
    - `GET /api/runs?state=` list (newest first) with recipe name, state, timestamps, `commercial_ok`.
    - `GET /api/runs/{id}` → DB fields + `status` (status.json) + `eval` (results.json) + `exports` (manifest) + `oom_hint` (`"CUDA out of memory: lower batch_size or enable gradient checkpointing"` when `oom`).
    - `POST /api/runs/{id}/cancel` → queued: `cancelled` immediately; running states: `cancel_requested=True` (202); terminal: 409.
    - `POST /api/runs/{id}/resume` → only `failed|cancelled`: `state=queued, resume=True, attempt+=1, cancel_requested=False, error=None, oom=False`; else 409; 507 if disk `block`.
    - `GET /api/runs/{id}/metrics?offset=0` → `{items, next_offset}`
    - `GET /api/runs/{id}/logs?tail=200` → `{stdout: [...], stderr: [...]}` (last N lines each)
    - `GET /api/runs/{id}/events` (SSE via `sse-starlette`): every 1 s emit `state` on change, `status` on change, `metric` per new line, `log` per new stdout/stderr line; close after emitting a terminal state.
    - `GET /api/runs/{id}/files/{path:path}` → file under `eval/` or `export/` only; 404 for anything resolving outside those.
  - Models:
    - `POST /api/runs/{id}/promote` → 201 new `ModelVersion` (`name` = recipe name, `number` = max for name + 1, `metrics` from eval results, `exports` from manifest); 409 unless `done`; **returns 200 with the existing version if this run is already promoted**.
    - `GET /api/models?task=`, `GET /api/models/{id}`
    - `POST /api/models/{id}/champion` → sets champion, unsets other champions of the same task.
    - `GET /api/models/{id}/exports/{index}` → `FileResponse` of that artifact.
  - System:
    - `read_gpu() -> GpuInfo | None` (`name, vram_used_mb, vram_total_mb, util_pct, temp_c` via `pynvml`; `None` on any error)
    - `GET /api/system` → `{gpu, lock: {holder_kind, holder_id, acquired_at}, disk: {free_gb, warn, block}, queue: {queued, running}}`

- [ ] **Step 1: Write the failing tests**

```python
async def test_create_run_freezes_recipe_and_sets_commercial_flag(client, seeded): ...
async def test_cancel_queued_is_immediate_and_terminal_is_409(client, seeded): ...
async def test_resume_only_from_failed_or_cancelled(client, seeded): ...
async def test_files_endpoint_blocks_traversal(client, seeded):
    assert (await client.get(f"/api/runs/{rid}/files/../config.json")).status_code == 404
    assert (await client.get(f"/api/runs/{rid}/files/eval/../../config.json")).status_code == 404
async def test_events_stream_ends_on_terminal_state(client, seeded): ...   # write status+metrics files, set state done, read stream to end
async def test_promote_twice_returns_same_version(client, done_run):       # Review Focus 5
    a = await client.post(f"/api/runs/{done_run}/promote"); b = await client.post(f"/api/runs/{done_run}/promote")
    assert a.status_code == 201 and b.status_code == 200 and a.json()["id"] == b.json()["id"]
async def test_one_champion_per_task(client, ...): ...
async def test_system_without_gpu(client, monkeypatch):
    monkeypatch.setattr("mlops_api.system.read_gpu", lambda: None)
    body = (await client.get("/api/system")).json()
    assert body["gpu"] is None and "free_gb" in body["disk"]
```

- [ ] **Step 2: Run to verify failure.** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run to verify pass.** → PASS.
- [ ] **Step 5: Commit** `feat(api): runs, model registry and system endpoints`

---

### Task 14: End-to-end pipeline test

**Files:**
- Test: `api/tests/test_e2e.py`

**Interfaces:**
- Consumes: everything above. No new production code unless the test finds a gap.

- [ ] **Step 1: Write the test.** Through the ASGI client and the worker ticks: tus-upload a 3-sample Bangla JSONL → ingest (`cpu_tick`) → save recipe (`backend: fake`, `task: llm-sft`, `export: ["dummy"]`, `hparams: {steps: 3}`) → create run → `gpu_tick` → `GET /api/runs/{id}` shows `done`, 3 loss metrics, eval `fake_score`, one export → promote → set champion → `GET /api/models?task=llm-sft` lists it as champion with `commercial_ok` matching provenance.
- [ ] **Step 2: Run it.** `uv run pytest api/tests/test_e2e.py -v` → PASS (fix any gap in its owning module, with a unit test there).
- [ ] **Step 3: Commit** `test: end-to-end pipeline with the fake backend`

---

### Task 15: Deployment: compose, Dockerfile, scripts

**Files:**
- Create: `docker-compose.yml`, `api/Dockerfile`, `web/Dockerfile` (multi-stage, `next build` then `next start`), `.env.example`, `scripts/setup_venvs.sh`, `scripts/check_vm.sh`, `scripts/dev.sh`, `README.md`

**Interfaces:**
- Produces:
  - Compose services: `postgres` (`postgres:16`, volume, port `5432` bound to `127.0.0.1`), `mlflow` (pin an exact MLflow release current at implementation time; backend store Postgres database `mlflow`, artifacts under `${DATA_ROOT}/mlflow`; port `5000` on `127.0.0.1`), `api` (runs `alembic upgrade head && uvicorn mlops_api.app:create_app --factory --host 0.0.0.0 --port 8000`; mounts `${DATA_ROOT}:${DATA_ROOT}`, `./backends:${REPO_ROOT}/backends:ro`, `./recipes:${REPO_ROOT}/recipes`; NVIDIA device reservation for NVML; port `8000` on `127.0.0.1`), `web` (port `3000` on `127.0.0.1`, `NEXT_PUBLIC_API_URL=http://localhost:8000`).
  - `.env.example`: `DATA_ROOT`, `REPO_ROOT`, `DATABASE_URL` (worker, `localhost`), `POSTGRES_PASSWORD`, `MLFLOW_TRACKING_URI=http://localhost:5000`, `WEB_ORIGIN`, `DISK_WARN_GB=20`, `DISK_MIN_GB=10`, `CANCEL_GRACE_S=60`.
  - `setup_venvs.sh`: `uv sync` at root; for each `backends/*/pyproject.toml` except `fake`, `uv sync --project backends/<name>`; prints each venv's python path.
  - `check_vm.sh`: checks `nvidia-smi`, `docker compose version`, `uv --version`, free disk on `DATA_ROOT` ≥ `DISK_MIN_GB`; prints `OK` or each failure.
  - `dev.sh`: `docker compose up -d --build`, waits for `GET /api/health`, then runs `uv run python -m mlops_worker` in the foreground; prints the tunnel command `ssh -L 3000:localhost:3000 -L 8000:localhost:8000 -L 5000:localhost:5000 <user>@<vm-ip>`.
  - `README.md`: setup on the VM, the tunnel, where data lives, how to add a backend (capabilities.json + `mlops_backend` CLI), and the fake backend's purpose.

- [ ] **Step 1: Write the files.**
- [ ] **Step 2: Verify locally where Docker exists:** `docker compose config` → valid; `bash -n scripts/*.sh` → no syntax errors.
- [ ] **Step 3: Commit** `build: compose stack, Dockerfiles and VM scripts`

---

### Task 16: Web scaffold, API client, layout and dashboard

**Files:**
- Create: `web/` via `npx create-next-app@15 web --ts --tailwind --app --src-dir --eslint --no-import-alias --use-npm`, then `npx shadcn@latest init`; `web/src/lib/api.ts`, `web/src/lib/format.ts`, `web/src/app/layout.tsx`, `web/src/components/nav.tsx`, `web/src/app/page.tsx` (dashboard), vitest config
- Test: `web/src/lib/format.test.ts`, `web/src/lib/api.test.ts`

**Interfaces:**
- Produces:
  - `api.ts`: `API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"`; `apiGet<T>(path)`, `apiPost<T>(path, body)`, `apiPatch<T>(path, body)`; non-2xx throws `ApiError(status, detail)` where `detail` is the FastAPI `detail` or the `errors` array joined by newline. Typed helpers for every route in Tasks 7, 8, 13 (`getSystem`, `listDatasets`, `getDataset`, `getVersion`, `getSamples`, `startIngest`, `getIngest`, `editTranscript`, `listBackends`, `validateRecipe`, `saveRecipe`, `listRecipes`, `getRecipe`, `cloneRecipe`, `createRun`, `listRuns`, `getRun`, `cancelRun`, `resumeRun`, `getLogs`, `promoteRun`, `listModels`, `setChampion`), and `runEventsUrl(id)`, `audioUrl(name, n, sampleId)`, `exportUrl(modelId, index)`.
  - `format.ts`: `formatBytes(n) -> string` ("1.5 GB"), `formatDuration(s) -> string` ("1h 02m", "45s"), `stateColor(state) -> string` (Tailwind class per run state).
  - Nav entries: Dashboard, Datasets, Recipes, Runs, Models. (Evaluate, Playground, Voice Call, Knowledge, Settings come in later sub-projects; do not add placeholder pages.)
  - Dashboard: GPU card (name, VRAM bar, util, temp, or "GPU not visible"), lock holder, disk card (amber when `warn`, red when `block`), queue counts, 10 latest runs, champions per task. Polls `/api/system` every 5 s.

- [ ] **Step 1: Write failing tests** for `formatBytes`, `formatDuration`, and `ApiError` (mock `fetch` returning `{status: 422, json: {errors: ["a","b"]}}` → `detail === "a\nb"`).
- [ ] **Step 2: Run** `npm run test` → FAIL.
- [ ] **Step 3: Implement** the files.
- [ ] **Step 4: Run** `npm run test && npm run typecheck && npm run build` → all pass.
- [ ] **Step 5: Commit** `feat(web): scaffold, API client, navigation and dashboard`

---

### Task 17: Datasets pages

**Files:**
- Create: `web/src/app/datasets/page.tsx`, `web/src/app/datasets/new/page.tsx`, `web/src/app/datasets/[name]/page.tsx`, `web/src/app/datasets/[name]/v/[n]/page.tsx`, `web/src/components/upload-form.tsx`, `web/src/components/report-view.tsx`, `web/src/components/sample-table.tsx`
- Test: `web/src/components/report-view.test.tsx`

**Interfaces:**
- Consumes: Task 16 API helpers.
- Produces:
  - Upload form: file picker, task select, name, eval-only switch, provenance fields (all required except notes; `commercial_ok` checkbox), tus upload with `tus-js-client` (`chunkSize` 8 MB, `endpoint` `${API_URL}/api/uploads`, resume via fingerprint), progress bar, then `startIngest` and polling `getIngest` every 2 s until `done|failed`; on done, navigate to the version page; on failed, show `ReportView`.
  - `ReportView({report})`: error and warning counts by code, expandable lists, sample IDs linking to filtered samples.
  - Version page: provenance card with license badge, split counts, `ReportView`, `SampleTable` with split and warning filters, paging (50), `<audio>` player per row for audio tasks, inline transcript edit (save → new version, navigate to it), eval-only toggle on the dataset page.

- [ ] **Step 1: Write failing test:** `ReportView` renders "2 errors" and groups warnings by code.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npm run test && npm run typecheck && npm run build` → pass.
- [ ] **Step 5: Commit** `feat(web): dataset upload, report and sample browser`

---

### Task 18: Recipes pages

**Files:**
- Create: `web/src/app/recipes/page.tsx`, `web/src/app/recipes/new/page.tsx`, `web/src/app/recipes/[name]/page.tsx`, `web/src/components/recipe-editor.tsx`, `web/src/components/schema-fields.tsx`
- Test: `web/src/components/schema-fields.test.tsx`

**Interfaces:**
- Produces:
  - `SchemaFields({schema, value, onChange})`: renders a JSON Schema `object` recursively; `integer`/`number` → number input with `minimum`/`maximum`; `string` → text input or select when `enum`; `boolean` → switch; shows `description` as help text. Unknown types render a read-only JSON preview.
  - `RecipeEditor`: tabs "Form" and "YAML" kept in sync (`yaml` package). Form: name, task select, backend select filtered by task (from `/api/backends`), method type select from the capability's `methods`, base model text, base model license + commercial checkbox, dataset selects (train: non-eval-only datasets of the task with version; eval: any of the task), export multi-select from `export_formats`, then `SchemaFields` for the backend schema. Calls `validateRecipe` 500 ms after the last change and lists errors; Save disabled while errors exist.
  - Recipe page: versions list with frozen badge, YAML view, Clone, "Start run" button (calls `createRun`, navigates to the run).

- [ ] **Step 1: Write failing tests:** `SchemaFields` renders a number input for `{"type":"integer","minimum":1}` and a select for an `enum`, and emits the updated object on change.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npm run test && npm run typecheck && npm run build` → pass.
- [ ] **Step 5: Commit** `feat(web): recipe editor with schema form and YAML view`

---

### Task 19: Runs pages

**Files:**
- Create: `web/src/app/runs/page.tsx`, `web/src/app/runs/[id]/page.tsx`, `web/src/components/metric-charts.tsx`, `web/src/components/log-view.tsx`, `web/src/components/phase-timeline.tsx`, `web/src/lib/metrics.ts`
- Test: `web/src/lib/metrics.test.ts`

**Interfaces:**
- Produces:
  - `metricsToSeries(items: Metric[]) -> Record<string, {step: number; value: number}[]>` keyed `"<split>/<name>"`, sorted by step, last value wins for duplicate steps (resumed runs).
  - Runs list: table with state badge, recipe, backend, commercial badge, started/duration; state filter.
  - Run detail: header with state, attempt, commercial badge, Cancel (running/queued) and Resume (failed/cancelled) buttons; `PhaseTimeline` (four phases, current highlighted, from status.json); progress (`step/total_steps`, ETA); `MetricCharts` (one recharts line chart per series); `LogView` (stdout/stderr tabs, auto-scroll, last 500 lines kept); `oom_hint` banner; eval metrics table; exports list with download links; Promote button when `done`. Live updates from `EventSource(runEventsUrl(id))`; initial state from `getRun`, `getLogs`, metrics endpoint.

- [ ] **Step 1: Write failing test** for `metricsToSeries` including duplicate steps after resume.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npm run test && npm run typecheck && npm run build` → pass.
- [ ] **Step 5: Commit** `feat(web): runs list and live run detail`

---

### Task 20: Models page and VM smoke check

**Files:**
- Create: `web/src/app/models/page.tsx`, `web/src/app/models/[id]/page.tsx`
- Modify: `README.md` (smoke-check section)

**Interfaces:**
- Produces: models table grouped by task (name, version, base model, key metrics, commercial badge, champion star); detail page with metrics, source run link, exports with download links, "Set champion" button.

- [ ] **Step 1: Implement** the pages.
- [ ] **Step 2: Run** `npm run test && npm run typecheck && npm run build` → pass.
- [ ] **Step 3: Commit** `feat(web): model registry pages`
- [ ] **Step 4: VM smoke check (the user runs this; report the results):** on the VM, `scripts/check_vm.sh` → `OK`; `scripts/setup_venvs.sh`; `cp .env.example .env` and set paths; `scripts/dev.sh`; on the PC, open the tunnel and `http://localhost:3000`; upload a small Bangla JSONL, create a `fake` recipe with `steps: 20, step_sleep_s: 0.5`, start a run, watch live metrics and logs, cancel it, resume it, let it finish, promote it, set champion; confirm the dashboard shows GPU and lock holder and the run appears in MLflow at `http://localhost:5000`.
- [ ] **Step 5: Commit** README changes `docs: VM smoke check for the core platform`
