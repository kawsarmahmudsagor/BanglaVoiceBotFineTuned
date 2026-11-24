# Speech Dataset Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the curation CLI (`tools/dataprep`) that turns Bengali.AI Speech and the stage-B voice recordings into platform datasets, plus a local-path ingest so 60–140 GB datasets enter the platform without a zip upload.

**Architecture:** `tools/dataprep` is a standalone uv project with its own venv. Each step reads a source's work directory (`var/dataprep/<source>/`) and writes one Parquet file, resumably. `select` applies a committed YAML profile; `export` writes the platform's existing upload layout (`metadata.csv` + `audio/` or `wavs/`, optional `split` column). The backend gains `POST /api/datasets/ingest-local`, which runs the existing ingest job on a directory under `import_root`, hard-linking files into the version.

**Tech Stack:** Python 3.12, uv, pandas/pyarrow, librosa, soundfile, pyloudnorm, soxr, torchmetrics DNSMOS, brouhaha, speechbrain ECAPA, scikit-learn, transformers, FastAPI + SQLAlchemy + Alembic, Next.js + vitest.

**Spec:** `docs/superpowers/specs/2026-11-24-speech-dataset-preparation-design.md`. Related: `docs/superpowers/specs/2026-11-23-training-backends-design.md` (§4 prepare contract, §15 splits, §16 CosyVoice).

## Context: the parallel training-backends session

Another session ("Dataset Structure Automation") is implementing the training-backends plan on `feat/training-backends` in the main checkout. It is at Task 20 and keeps committing there. Facts this plan relies on:

- Upload layout (already on that branch): `metadata.csv` with an optional `split` column (`train`, `valid`/`val`/`dev`, `test`), **or** `metadata_train.csv` / `metadata_valid.csv` / `metadata_test.csv`. Mixing both is an ingest error. Reading is `utf-8-sig`, blank rows skipped. `split` is not a reserved column.
- `ingest()` finds metadata at the root or one directory below. It records `given_splits` and `split_source`.
- Splits are decided per run (`app.core.splits`). A dataset with a `split` column gets `split_source = "dataset"`.
- Training backends read audio **by reference** from `<version>/files/<file>`. Version files must stay in place and unchanged, otherwise prepared caches go stale.
- Its next migration would be `0004`. **This plan claims `0004`**; Task 15 tells that session.
- `GET /api/datasets/{name}/versions/{n}` response shape (`split_source`, `given_splits`, `splits`) must not change.
- Its Task 34 adds `app/core/tts_refs.py` and the CosyVoice converter. Spec §13 corrections go to that session (Task 15); this plan does not edit its spec, plan or backend code.

## Global Constraints

- Work on branch `feat/speech-dataprep`, in a worktree (`superpowers:using-git-worktrees`). The user merges `feat/training-backends` into local `main` first; then local `main` is merged into `feat/speech-dataprep` so it has every update. Never commit on `feat/training-backends`. Implementation starts only when the user says so.
- Everything must install and run on the Linux VM (Ubuntu, RTX 5090, CUDA 12.8+). Lockfiles resolve with `environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]`.
- Torch is `2.7.1` from the `pytorch-cu128` index in every venv.
- `tools/dataprep` imports only pure-Python `app.core` modules (`app.core.text`, `app.core.metrics`, `app.core.splits`). Never `app.routers`, `app.db` or `workers`.
- Curated `text` is the cleaned text (spec §7.1), never a `bn_normalize` output.
- CER and WER always go through `bn_normalize(mode="asr")` on both sides. Never Whisper's `BasicTextNormalizer`.
- Exports use the existing upload layout and pass `app.datasets.validate_audio.validate_audio` with zero errors.
- A file in an export directory is never rewritten in place: write to `<name>.tmp-<pid>` then `os.replace`. Hard-linked dataset versions depend on this.
- Bengali.AI datasets are ingested with `commercial_ok: false` until its license is checked.
- GPU steps are verified on the VM by `tools/dataprep/scripts/vm_smoke.sh`. CPU tests never download models or need CUDA: scorer callables are injected.
- Test commands (local): `PYTHONPATH=tools/dataprep uv run --project backend --with pytest --with pandas --with pyyaml python -m pytest tools/dataprep/tests -q` and `uv run --project backend pytest -q`. Frontend: `cd frontend && npm test`.

## Review Focus

1. **A hard-linked version file is changed after ingest.** Re-exporting a dataset or editing a file in `var/import/` must not change an ingested version. Pinned by: Task 5 `test_export_replaces_not_rewrites` (new inode), Task 7 `test_ingest_local_files_read_only`.
2. **One bad source file mid-run (undecodable MP3, odd Unicode name).** The step records an error for that row and continues; a rerun does not retry it unless `--retry-errors`. Pinned by: Task 1 `test_step_records_row_errors`, Task 3 `test_audio_stats_undecodable`.
3. **A step file from an older step version or different parameters.** `select` must refuse to use it instead of silently mixing results. Pinned by: Task 1 `test_stale_step_file_refused`.
4. **A profile that removes every row, or one split.** `select` fails with the filter that emptied it; `export` refuses an empty selection. Pinned by: Task 4 `test_select_empty_result_names_filter`.
5. **Text that is empty after `bn_normalize(mode="asr")`** (punctuation-only labels). It must be dropped with a reason, not crash CER or `cps`. Pinned by: Task 2 `test_text_flags_empty_after_normalize`, Task 12 `test_cer_skips_empty_reference`.

---

## File Structure

```
tools/dataprep/
  pyproject.toml                 extras: cosyvoice (exists), score, labeller; uv conflicts between them
  uv.lock
  dataprep/__main__.py           subcommand registry (exists; extended per task)
  dataprep/workdir.py            WorkDir, StepFile: paths, versioned Parquet IO, resume
  dataprep/pool.py               run_rows(): process pool, per-row error capture, progress
  dataprep/sources/bengaliai.py  import: train.csv + train_mp3s/ -> clips.parquet
  dataprep/steps/text.py         clean(), flags
  dataprep/steps/audio_stats.py
  dataprep/steps/dnsmos.py
  dataprep/steps/brouhaha.py
  dataprep/steps/transcribe.py   labellers: whisper-hf, nemo
  dataprep/steps/speakers.py
  dataprep/profiles.py           Profile model, load_profile()
  dataprep/select.py             apply filters, report, char audit, overlap
  dataprep/export.py             ASR + TTS export, curation.json
  dataprep/audio_proc.py         trim, loudness, resample (TTS)
  dataprep/listen.py             listening sheet HTML
  dataprep/check_voice.py        stage-B QA
  dataprep/check_eval_text.py    domain-eval-tts overlap check
  dataprep/roundtrip_cosyvoice.py  (exists)
  profiles/asr-v1.yaml asr-v2.yaml tts-loose.yaml tts-strict.yaml voice.yaml
  data/phonetic_coverage.txt     vowel signs and conjuncts for check-voice
  scripts/setup_cosyvoice.sh     (exists)
  scripts/vm_smoke.sh
  tests/fixtures.py              make_clip_dir(tmp, rows) -> writes wavs + metadata
  tests/test_*.py

backend/
  alembic/versions/0004_ingest_source_path.py
  app/settings.py                + import_root
  app/models.py                  Ingest.upload_id nullable, + Ingest.source_path
  app/datasets/ingest.py         + ingest_dir()
  app/datasets/validate_audio.py + workers param (process pool)
  app/datasets/report.py         + warning cap
  app/datasets/versions.py       run_ingest: source_path branch
  app/routers/datasets.py        + POST /datasets/ingest-local, GET /datasets/import-dirs
  tests/app/test_ingest_local.py
frontend/src/
  lib/api.ts                     + ingestLocal(), importDirs()
  components/server-path-form.tsx (+ .test.tsx)
  components/upload-form.tsx     source toggle: upload | server path
```

---

### Task 1: Scaffold, work directory and Bengali.AI import

Commits the existing `tools/dataprep` files (round-trip gate, setup script, lockfile) and the spec together with the first real step.

**Files:**
- Create: `tools/dataprep/dataprep/workdir.py`, `dataprep/pool.py`, `dataprep/sources/__init__.py`, `dataprep/sources/bengaliai.py`, `tests/fixtures.py`, `tests/test_workdir.py`, `tests/test_import_bengaliai.py`
- Modify: `tools/dataprep/pyproject.toml` (base deps `pandas`, `pyarrow`, `pyyaml`), `dataprep/__main__.py`
- Commit as is: `docs/superpowers/specs/2026-11-24-speech-dataset-preparation-design.md`, `tools/dataprep/**` already present

**Interfaces:**
- Produces:
  - `WorkDir(root: Path)` with `.clips_path`, `.step_path(name: str) -> Path`, `.selection_path(profile: str) -> Path`, `.export_dir(dataset: str) -> Path`. Default root `var/dataprep/<source>`.
  - `StepFile.write(path, df: pd.DataFrame, *, step: str, version: int, params: dict) -> None` (metadata stored in Parquet schema metadata as JSON under key `dataprep`).
  - `StepFile.read(path, *, step: str, version: int, params: dict | None = None) -> pd.DataFrame` raises `StaleStepError` when step/version differ, or params differ when given.
  - `StepFile.done_ids(path) -> set[str]` ids present with `error` null.
  - `run_rows(fn: Callable[[dict], dict], rows: list[dict], *, workers: int, desc: str) -> list[dict]`: each result has `id` and `error: str | None`; an exception in `fn` becomes `{"id": row["id"], "error": "<Type>: <msg>"}`.
  - `import_bengaliai(src: Path, wd: WorkDir) -> int` writes `clips.parquet` columns `id, src_path, text_raw, src_split` (`train`→`train`, `valid`→`val`, via `app.core.splits.normalize_split`).
  - Step CLI convention used by every later step: `python -m dataprep <step> --work <dir> [--workers N] [--force] [--retry-errors]`.

- [ ] **Step 1: Write failing tests**

```python
def test_stepfile_roundtrip_and_stale(tmp_path):
    p = tmp_path / "s.parquet"
    StepFile.write(p, pd.DataFrame({"id": ["a"], "x": [1], "error": [None]}), step="s", version=1, params={"k": 1})
    assert StepFile.read(p, step="s", version=1, params={"k": 1}).x.tolist() == [1]
    with pytest.raises(StaleStepError): StepFile.read(p, step="s", version=2)
    with pytest.raises(StaleStepError): StepFile.read(p, step="s", version=1, params={"k": 2})

def test_stale_step_file_refused(tmp_path): ...  # same via the select loader, Task 4 reuses

def test_step_records_row_errors():
    out = run_rows(lambda r: 1 / 0 if r["id"] == "b" else {"id": r["id"], "v": 1}, [{"id": "a"}, {"id": "b"}], workers=1, desc="t")
    assert {o["id"]: o["error"] for o in out} == {"a": None, "b": "ZeroDivisionError: division by zero"}

def test_import_bengaliai(tmp_path):
    # train.csv rows: id,sentence,split with splits train/valid; train_mp3s/<id>.mp3 present for both
    n = import_bengaliai(src, WorkDir(tmp_path / "wd"))
    df = pd.read_parquet(tmp_path / "wd" / "clips.parquet")
    assert n == 2 and sorted(df.src_split) == ["train", "val"]
    assert df.src_path.map(lambda s: Path(s).is_file()).all()

def test_import_bengaliai_missing_audio_is_reported(tmp_path): ...  # row without mp3 -> not imported, count printed

def test_import_bengaliai_wrong_dir_names_expected_files(tmp_path):
    with pytest.raises(SystemExit, match="train.csv"): main(["import", "bengaliai", "--src", str(tmp_path), "--work", str(tmp_path / "wd")])
```

- [ ] **Step 2: Run, verify they fail** (`ImportError: workdir`).
- [ ] **Step 3: Implement** `workdir.py`, `pool.py` (`concurrent.futures.ProcessPoolExecutor`, `chunksize=64`, `workers=1` runs inline for tests), `sources/bengaliai.py`, and register `import bengaliai --src <dir> --work <dir>`. `--src` defaults to `$BENGALIAI_SRC`, else `~/.cache/kagglehub/competitions/bengaliai-speech` (where the VM copy already is: `/home/xr23/.cache/kagglehub/competitions/bengaliai-speech`). Files are read in place; `src_path` is absolute (resolved), nothing is copied. A `--src` without `train.csv` and `train_mp3s/` exits with a message naming both.
- [ ] **Step 4: Run all dataprep tests, verify pass** (including the 11 existing round-trip tests).
- [ ] **Step 5: `uv lock --project tools/dataprep`**, verify it resolves.
- [ ] **Step 6: Commit** `feat(dataprep): scaffold, work dir, bengaliai import, cosyvoice round-trip gate` (include the spec and existing files).

### Task 2: `text` step

**Files:** Create `dataprep/steps/text.py`, `tests/test_text.py`. Modify pyproject (base dep `bnunicodenormalizer`; first check its license on PyPI and record it in the commit message; if not permissive, stop and ask).

**Interfaces:**
- Consumes: `WorkDir`, `StepFile` (Task 1).
- Produces: `clean(text: str) -> str`; step file `text.parquet` (step `text`, version 1) columns `id, text_clean, has_latin: bool, has_digits: bool, n_chars: int, empty_asr: bool, error`. `n_chars` counts non-space characters of `bn_normalize(text_clean, mode="asr")`.

`clean` order (spec §7.1): `bnunicodenormalizer` per word, then NFC, then zero-width removal and danda/quote unification reused from `app.core.text` private helpers (`_strip_zero_width`; import, do not copy), then whitespace collapse and strip.

- [ ] **Step 1: Failing tests:**
  - `test_clean_table`: parametrized pairs. Cover decomposed vs precomposed `য়` giving identical output, stray ZWNJ outside a conjunct removed, `|` → `।`, double spaces collapsed, Latin and digits kept unchanged.
  - `test_clean_idempotent`: `clean(clean(t)) == clean(t)` over the table inputs.
  - `test_clean_roundtrip_property`: `bn_normalize(clean(t), mode=m) == bn_normalize(t, mode=m)` for `m in ("asr", "tts")` over inputs without broken sequences.
  - `test_text_flags_empty_after_normalize`: `"।।"` gives `empty_asr=True`, `n_chars=0`.
- [ ] **Step 2: Verify fail. Step 3: Implement and register `text`. Step 4: Verify pass. Step 5: Commit** `feat(dataprep): text cleaning step`.

### Task 3: `audio-stats` step

**Files:** Create `dataprep/steps/audio_stats.py`, `tests/test_audio_stats.py`.

**Interfaces:**
- Produces: `audio_stats(path: Path) -> dict` with `dur, src_sr, clip_frac, silence_share, lufs`; step file `audio_stats.parquet` (version 1).
- Definitions (spec §6.1): `clip_frac` = share of samples with `abs >= 0.99`; `silence_share` = share of 20 ms frames with RMS below −50 dBFS; `lufs` via `pyloudnorm.Meter(sr)`, `None` for clips under 0.4 s. Decode with `librosa.load(path, sr=None, mono=True)`.

- [ ] **Step 1: Failing tests:**
  - `test_audio_stats_values`: 2 s sine at 0.5 amplitude, 22050 Hz, plus 1 s of zeros → `dur == 3.0 ± 0.01`, `src_sr == 22050`, `clip_frac == 0`, `silence_share == 1/3 ± 0.02`.
  - `test_audio_stats_clipped`: square wave at 1.0 → `clip_frac > 0.9`.
  - `test_audio_stats_undecodable`: a `.mp3` file containing text bytes → row has `error` set, other rows succeed.
  - `test_resume_skips_done`: run twice, second run computes 0 rows (count via an injected counter).
- [ ] **Steps 2–4** as usual. **Step 5: Commit** `feat(dataprep): audio-stats step`.

### Task 4: Profiles and `select`

**Files:** Create `dataprep/profiles.py`, `dataprep/select.py`, `profiles/*.yaml` (5 files), `tests/test_select.py`.

**Interfaces:**
- Consumes: step files from Tasks 2, 3 (and 9, 10, 12 when a profile references their columns).
- Produces:
  - `Profile` (pydantic): `name, task: Literal["asr","tts"], filters: list[Filter], drop_latin: bool, min_speaker_minutes: float | None, cer_labeller: str | None`.
  - `Filter` (one of): `Range(column, min=None, max=None)`, `Percentile(column, lo, hi)`, `Flag(column, keep: bool)`.
  - `load_profile(name_or_path) -> Profile` (bare names resolve to `tools/dataprep/profiles/<name>.yaml`).
  - `select(wd: WorkDir, profile: Profile) -> SelectResult(rows: pd.DataFrame, report: dict)`; writes `selections/<name>.parquet` and `selections/<name>.report.json`.
  - Report keys: `steps: [{filter, rows_before, rows_after, hours_before, hours_after}]`, `chars: [{char, codepoint, count, examples}]`, `unexpected_chars: [...]`, `overlap_dropped: int`.
  - `cps` is derived in `select` as `n_chars / dur`.
- Profile values: copy spec §6.3 exactly (durations, `clip_frac < 0.001`, `cps` percentiles, CER, DNSMOS, SNR, C50, `max_pause_s`, Latin, speaker minutes). Rows with any step `error` or `empty_asr` are always dropped first, as filter `"step_errors"`.
- Overlap (spec §7.3): drop `train` rows whose `bn_normalize(text_clean, mode="asr")` equals a `val` row's.
- Expected characters (spec §7.2): Bengali block U+0980–U+09FF, ASCII letters and digits, space, ZWJ/ZWNJ, and `।?,.!-'"`. Anything else is listed in `unexpected_chars`.

- [ ] **Step 1: Failing tests:**
  - `test_each_filter_removes_expected_rows`: a 10-row synthetic step set; each filter type drops exactly the planted rows; report counts match.
  - `test_select_empty_result_names_filter`: a profile with `dur.min = 100` → `SelectError` whose message contains `"dur"`.
  - `test_overlap_dropped_from_train_only`.
  - `test_unexpected_chars_listed`: a row containing `€` appears in `unexpected_chars` with its row id as an example.
  - `test_profile_missing_step_column`: `tts-strict` without `brouhaha.parquet` → error naming the missing step.
  - `test_shipped_profiles_load`: all five YAML files load and match the spec table values spot-checked (`tts-strict` `dnsmos_ovrl >= 3.2`, `asr-v2` CER `<= 0.5`).
- [ ] **Steps 2–4.** **Step 5: Commit** `feat(dataprep): filter profiles and select`.

### Task 5: ASR export

**Files:** Create `dataprep/export.py`, `tests/test_export_asr.py`.

**Interfaces:**
- Consumes: `select` output (Task 4), `clips.parquet`, `text.parquet`.
- Produces: `export(wd: WorkDir, profile: Profile, dataset: str, *, workers: int) -> Path` returning the export dir. ASR layout: `metadata.csv` (`file,text,split`), `audio/<id>.flac` (16 kHz mono via `soxr`), `curation.json` (spec §4.4 keys: `source, profile, step_versions, report, tool_commit, provenance`). The `provenance` object holds the fields for the ingest form, with `commercial_ok: false` for Bengali.AI.
- `curation.json` must not trip ingest. `validate_audio` only indexes audio extensions, so a JSON file at the root is safe; the test proves it.
- Re-export rule: files are written as `<name>.tmp-<pid>` then `os.replace`. Ids no longer selected are removed from the export dir. `export` refuses an empty selection.

- [ ] **Step 1: Failing tests:**
  - `test_export_passes_validate_audio`: export of a 5-clip fixture; `validate_audio(dir, "asr")` returns no errors; `given` splits match.
  - `test_export_replaces_not_rewrites`: hard-link an exported file elsewhere, re-export, assert the link still holds the old bytes and the export file has a new inode.
  - `test_export_removes_deselected`.
  - `test_export_refuses_empty_selection`.
- [ ] **Steps 2–4.** **Step 5: Commit** `feat(dataprep): asr export`.

### Task 6: Backend — warning cap and parallel validation

Touches files owned by the training-backends work; keep the change additive. Task 15 tells that session.

**Files:** Modify `backend/app/datasets/report.py`, `backend/app/datasets/validate_audio.py`, `backend/app/datasets/ingest.py` (pass `workers`). Test: `backend/tests/app/datasets/test_validate_audio.py`.

**Interfaces:**
- Produces:
  - `Report.add_warning(issue: Issue, cap: int = 1000) -> None` keeps at most `cap` stored per `issue.code`, and always increments `stats["warning_counts"][code]`. `_audio_warnings` uses it.
  - `validate_audio(root, task, *, workers: int = 1)`: decoding and `_audio_warnings` run per file in a `ProcessPoolExecutor` when `workers > 1`. Row order and every output field are identical to `workers=1`.
  - `ingest(..., workers: int = 1)`; `run_ingest` passes `settings.ingest_workers` (new setting, default `os.cpu_count() // 2`, min 1).

- [ ] **Step 1: Failing tests:**
  - `test_warning_cap`: 1200 clipped clips (tiny, 0.1 s; build via the existing `wav()` helper), `cap=1000` → 1000 stored `clipping` warnings, `warning_counts["clipping"] == 1200`.
  - `test_parallel_matches_serial`: the same fixture with `workers=1` and `workers=4` gives equal `rows` and equal reports.
- [ ] **Steps 2–4** with `uv run --project backend pytest tests/app/datasets -q`. **Step 5: Commit** `feat(datasets): parallel audio validation and warning cap`.

### Task 7: Backend — local-path ingest

**Files:** Create `backend/alembic/versions/0004_ingest_source_path.py`, `backend/tests/app/test_ingest_local.py`. Modify `app/settings.py`, `app/models.py`, `app/datasets/ingest.py`, `app/datasets/versions.py`, `app/routers/datasets.py`.

**Interfaces:**
- Consumes: `ingest(..., workers)` (Task 6).
- Produces:
  - Setting `import_root: Path` (default `data_root / "import"`).
  - `Ingest.upload_id` nullable; new `Ingest.source_path: str | None`. The migration uses `op.batch_alter_table` (SQLite) and has a downgrade.
  - `ingest_dir(src: Path, task: str, dest_root: Path, disk_min_gb: float = 0, workers: int = 1) -> IngestResult`. It validates `src` in place (no extraction), then for each row links `src/<file>` to `<version>/files/<file>` with `os.link`, falling back to `shutil.copy2` on `OSError` (cross-device). Every version file is then made read-only (`chmod 0o444`). The disk check counts only bytes that must be copied.
  - `resolve_import_path(raw: str, import_root: Path) -> Path` raises `ImportPathError` when the resolved path (after `Path.resolve()`, so symlinks are followed) is not inside the resolved `import_root`, or is not a directory.
  - `POST /api/datasets/ingest-local` body `{path, name, task, eval_only, provenance}` (same `Provenance`, `SLUG`, task checks as `IngestRequest`). Responses: `202 {"ingest_id"}`, `403` for `ImportPathError`, `404` when the directory is missing, `409` for a task mismatch with an existing dataset.
  - `GET /api/datasets/import-dirs` → `{"items": [{"path": "<relative to import_root>", "task_hint": "asr"|"tts"|null, "has_curation": bool}]}`. It lists directories up to 2 levels deep that contain `metadata.csv` or `metadata_<split>.csv`. `task_hint` comes from `curation.json` when present.
  - `run_ingest`: when `row.source_path` is set, call `ingest_dir` instead of the upload path; never delete the source.
- `get_version` and its response shape are untouched.

- [ ] **Step 1: Failing tests** (`test_ingest_local.py`, using the existing API test client fixtures):
  - `test_ingest_local_ok`: fixture dir under a tmp `import_root` → 202; after `run_ingest` the version exists with `split_source == "dataset"`.
  - `test_ingest_local_rejects_outside_root`, `test_ingest_local_rejects_dotdot`, `test_ingest_local_rejects_symlink_escape` (skip on Windows without symlink rights) → 403.
  - `test_ingest_local_hardlinks_same_fs`: `os.stat(version_file).st_ino == os.stat(src_file).st_ino`; the source dir is unchanged after ingest.
  - `test_ingest_local_files_read_only`: version files have no write bits.
  - `test_ingest_local_copy_fallback`: monkeypatch `os.link` to raise `OSError(18)` → files copied, ingest succeeds.
  - `test_import_dirs_lists_dataset_dirs`.
  - `test_upload_ingest_unchanged`: the existing upload ingest test still passes (run the old suite).
- [ ] **Step 2: Verify fail. Step 3: Implement.** Run `uv run --project backend alembic upgrade head` and `alembic downgrade -1` on a scratch SQLite DB to check the migration.
- [ ] **Step 4: Full backend suite passes.** **Step 5: Commit** `feat(datasets): local-path ingest with hard links (migration 0004)`.

### Task 8: Frontend — "From server path"

**Files:** Create `frontend/src/components/server-path-form.tsx`, `server-path-form.test.tsx`. Modify `frontend/src/lib/api.ts`, `frontend/src/components/upload-form.tsx`.

**Interfaces:**
- Consumes: the Task 7 endpoints.
- Produces: `api.importDirs(): Promise<{items: ImportDir[]}>`, `api.ingestLocal(body): Promise<{ingest_id: number}>`. `ServerPathForm({ defaultTask, onStatus })` reuses the upload form's provenance fields and ingest polling (extract the shared part into a component inside `upload-form.tsx` if needed; do not duplicate it).
- UI: a two-option toggle at the top of the upload form, "Upload file" | "From server path". Server path shows a select of `importDirs` items plus a free-text path field. Choosing an item with `task_hint` preselects the task.

- [ ] **Step 1: Failing tests:** renders the listed dirs; choosing one with `task_hint: "tts"` sets the task; submit calls `ingestLocal` with the relative path and the provenance; a 403 shows the API's error message.
- [ ] **Steps 2–4** with `npm test`. **Step 5: Commit** `feat(web): ingest from server path`.

### Task 9: `dnsmos`, `brouhaha` and `listen`

**Files:** Create `dataprep/steps/dnsmos.py`, `dataprep/steps/brouhaha.py`, `dataprep/listen.py`, `tests/test_scores.py`, `tests/test_listen.py`. Modify pyproject: extra `score` = `torch`, `torchaudio` (cu128), `torchmetrics[audio]>=1.4`, `onnxruntime-gpu`, `brouhaha` (git source per its README; pin a commit), `speechbrain`, `scikit-learn`.

**Interfaces:**
- Produces:
  - `score_dnsmos(y16k: np.ndarray) -> dict(dnsmos_p808, dnsmos_sig, dnsmos_bak, dnsmos_ovrl)`; step `dnsmos` (version 1).
  - `score_brouhaha(path) -> dict(snr_db, c50_db, max_pause_s, speech_segments: list[[start, end]])`; step `brouhaha` (version 1). Options: `--only-profile <name>` restricts rows to those passing that profile's non-brouhaha filters (spec §5.2 step 2).
  - Each step module takes an injectable `scorer` for tests; the default loads the real model lazily.
  - `listen(wd, profile, *, column="dnsmos_ovrl", per_band=10, bands=[0,2.5,2.8,3.0,3.2,3.5,5]) -> Path` writes `listen/<profile>.html` with relative audio links and the clip's scores.

- [ ] **Step 1: Failing tests:** step plumbing with a fake scorer (rows written, resume, errors recorded); `max_pause_s` derived correctly from synthetic segments `[[0,1],[2.5,4]]` → `1.5`; listening sheet has `per_band` rows per band and escapes text.
- [ ] **Steps 2–4.** **Step 5: Commit** `feat(dataprep): dnsmos, brouhaha and listening sheet`.

### Task 10: `speakers` step

**Files:** Create `dataprep/steps/speakers.py`, `tests/test_speakers.py`.

**Interfaces:**
- Produces: `cluster(embeddings: np.ndarray, threshold: float) -> np.ndarray[int]` (agglomerative clustering, cosine distance, `linkage="average"`, `distance_threshold=threshold`); step `speakers` (version 1) columns `id, speaker` with names `bai-<n>`, numbered by cluster size descending. The embedder is speechbrain ECAPA (`speechbrain/spkrec-ecapa-voxceleb`, 192-d), injectable. Default `--threshold 0.35`, recorded in params.
- `select` uses `min_speaker_minutes` with `speakers.parquet` + `dur`.

- [ ] **Step 1: Failing tests:** three synthetic Gaussian clusters of 192-d unit vectors → 3 clusters; naming by size; `min_speaker_minutes` filter drops the small cluster (Task 4 select integration).
- [ ] **Steps 2–4.** **Step 5: Commit** `feat(dataprep): pseudo-speaker clustering`.

### Task 11: TTS export processing

**Files:** Create `dataprep/audio_proc.py`, `tests/test_audio_proc.py`. Modify `dataprep/export.py`, `tests/test_export_tts.py`.

**Interfaces:**
- Consumes: `brouhaha` segments (Task 9), `speakers` (Task 10).
- Produces:
  - `process_tts(y, sr, segments, *, pad_s=0.15, target_lufs=-23.0, peak_dbtp=-1.0, out_sr=24000) -> np.ndarray` in spec §8.3 order: mono, trim to the first/last segment ± pad, loudness-normalize, true-peak limit (4× oversampled peak estimate with `soxr`, then gain), resample.
  - TTS export layout: `metadata.csv` (`file,text,speaker,split`), `wavs/<id>.wav` PCM_16 24 kHz, `curation.json` with `band_limited` count (source rate < 24000). TTS export decodes `clips.src_path` (the original MP3), never ASR export audio.
  - Speaker for the `voice` profile comes from `--speaker <name>` (single speaker).

- [ ] **Step 1: Failing tests:** trim keeps 150 ms around segments; output loudness −23 ± 1 LUFS; true peak ≤ −1 dBTP; output rate 24000; `band_limited` counted for a 16 kHz source; the TTS export passes `validate_audio(dir, "tts")` with no errors.
- [ ] **Steps 2–4.** **Step 5: Commit** `feat(dataprep): tts export with trim, loudness and resample`.

### Task 12: `transcribe` step and labellers

**Files:** Create `dataprep/steps/transcribe.py`, `tests/test_transcribe.py`. Modify pyproject: extra `labeller` = `transformers>=4.53`, `accelerate`, `nemo_toolkit[asr]`, torch cu128. Declare `[tool.uv] conflicts = [[{extra = "cosyvoice"}, {extra = "labeller"}]]` (CosyVoice needs `transformers==4.51.3`).

**Interfaces:**
- Produces:
  - `Labeller` protocol: `transcribe(paths: list[Path]) -> list[str]`.
  - `--labeller whisper-hf:<hf id or local dir>` (transformers pipeline, `generate_kwargs={"language": "bengali", "task": "transcribe"}`, batched) and `--labeller nemo:<hf id or .nemo path>` (`nemo_asr.models.ASRModel.from_pretrained` or `restore_from`). BanglaConformer is `nemo:bengaliAI/BanglaConformer`.
  - Step file `cer_<slug>.parquet`, where slug = labeller string with non-alphanumerics replaced by `-`. Columns `id, hyp, cer, error`. `cer` uses `app.core.metrics.cer` over `bn_normalize(mode="asr")` texts.
  - Profiles reference it via `cer_labeller`; `select` reads column `cer` from that file.
  - Loading a platform run's exported model dir is the `whisper-hf:<dir>` form.

- [ ] **Step 1: Failing tests** with a fake labeller: CER values for known pairs; `test_cer_skips_empty_reference` (row with `empty_asr` gets `error="empty reference"`, not a ZeroDivision); slug naming; resume.
- [ ] **Steps 2–4.** **Step 5: Commit** `feat(dataprep): transcribe step with whisper and nemo labellers`.

### Task 13: Stage-B voice: import, `check-voice`, export

**Files:** Create `dataprep/sources/voice.py`, `dataprep/check_voice.py`, `data/phonetic_coverage.txt`, `tests/test_check_voice.py`.

**Interfaces:**
- Consumes: Tasks 1–5, 9, 11, 12 steps (the voice set is a source like any other).
- Produces:
  - `import voice --src <dir with metadata.csv + wavs/> --work <dir>`: `clips.parquet` with `src_split` empty; optional `category` column kept.
  - `check-voice --work <dir> --labeller <spec>` runs `text`, `audio-stats`, `dnsmos`, `brouhaha`, `transcribe`, then writes `check_voice.html` plus `check_voice.json`. Flags use the `voice` profile thresholds (CER > 0.02, `dnsmos_ovrl` < 3.5, `snr_db` < 35, `max_pause_s` > 1.0, clipping). Per set: hours, sample rate and bit depth consistency (expect 48000 Hz, 24-bit; read via `soundfile.info`), speech rate (`n_chars/dur`) mean per session (`session` column if present, else per recording date from the file mtime), and phonetic coverage.
  - Coverage: each line of `phonetic_coverage.txt` is one target (a vowel sign or a conjunct); count its occurrences in `text_clean`; report targets below 20.
  - Test split (spec §9.4): `export --profile voice --test-count 150` assigns `split=test` to 150 takes. With a `category` column it picks round-robin across categories in hash order; otherwise hash order. Other rows get no split (platform default ratios).

- [ ] **Step 1: Failing tests:** flags raised for planted bad takes; coverage counts; a 44.1 kHz take flagged as inconsistent; test split takes exactly 150 rows and covers every category in a 3-category fixture.
- [ ] **Steps 2–4.** **Step 5: Commit** `feat(dataprep): stage-B voice import, check-voice and test split`.

### Task 14: Domain eval text check and VM smoke

**Files:** Create `dataprep/check_eval_text.py`, `tests/test_check_eval_text.py`, `scripts/vm_smoke.sh`, `tools/dataprep/README.md`.

**Interfaces:**
- Produces:
  - `check-eval-text --sentences <txt> --against <work dir> [--against ...]`: reports sentences whose `bn_normalize(mode="asr")` form matches any `text_clean` in the given work dirs, and exits 1 if any match (spec §10.2).
  - `vm_smoke.sh`: builds a 20-clip fixture from the first 20 Bengali.AI rows of the kagglehub cache (`${BENGALIAI_SRC:-$HOME/.cache/kagglehub/competitions/bengaliai-speech}`), runs every step including GPU steps with real models, runs `select` for each profile with thresholds relaxed via `--override` (add `--override column.min=value` to `select` in this task if Task 4 did not), exports, and runs `ingest_dir` through the API on the export. Prints `SMOKE OK` at the end.
  - README: the command order from spec §5 (STT, TTS stage A, stage B, gate), with each command line. Start from the kagglehub cache path and say not to delete the cache before the TTS exports are ingested.

- [ ] **Step 1: Failing test:** a sentence present in a work dir is reported; exit code 1; no match gives 0.
- [ ] **Steps 2–4.** **Step 5: Run `vm_smoke.sh` on the VM**; expected last line `SMOKE OK`. **Step 6: Commit** `feat(dataprep): eval text check, vm smoke, readme`.

### Task 15: Hand-offs to the training-backends session

No code. Send one message to the "Dataset Structure Automation" session (SendMessage) and paste it into the final summary for the user:

1. Migration `0004_ingest_source_path` is taken by this branch; use `0005` for its next one.
2. Additive changes to `report.py` (`add_warning`, `warning_counts`), `validate_audio.py` (`workers`), `ingest.py` (`ingest_dir`, `workers`), `versions.py` (`source_path` branch) and two new routes in `routers/datasets.py`. `get_version` is unchanged.
3. Version files may now be hard links and are read-only (`0o444`). Backends must not write into `<version>/files/`.
4. The spec §13 corrections for its §16 CosyVoice backend: default `components` `["llm"]`; the `"You are a helpful assistant.<|endofprompt|>"` text prefix in the converter; `use_spk_embedding: True` only for stage-B SFT; `transformers < 4.53`; the upstream stage-6 path bug; about 1000 h guidance.
5. The spec §14 ASR requirements: no telephony augmentation; noise 10–30 dB SNR on 20–30 % of clips; mild reverb; Opus only for WebRTC; Qwen3-ASR fixed language label.
6. The round-trip gate result (`gate.json`) once run, since it decides whether CosyVoice 3 stays the main TTS track.

- [ ] **Step 1: Send the message.** **Step 2: Merge the latest local `main` into `feat/speech-dataprep`, rerun all three test suites, and report results.**

---

## Execution notes

- Tasks 1–5 deliver the STT path end to end; with Tasks 6–8 the first STT dataset can be ingested and trained. That is the first useful checkpoint.
- Run the CosyVoice round-trip gate on the VM (spec §12) in parallel with Task 1: it is already built and its result decides how much of Tasks 9–11 matters for CosyVoice versus F5.
- GPU steps (Tasks 9, 10, 12) have CPU tests only for plumbing; their real behaviour is proven by `vm_smoke.sh` in Task 14.
