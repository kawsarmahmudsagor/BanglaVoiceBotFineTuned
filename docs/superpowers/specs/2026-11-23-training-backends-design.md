# Training backends (prepare, train, eval, export) — design

Date: 2026-11-23. Branch: `feat/training-backends`. Parent spec: `2026-11-23-bangla-voice-mlops-design.md`.

## 1. Goal

Make LLM, STT and TTS models trainable for real. Each task keeps one upload format (parent spec §4.2). Before training, every run converts the stored dataset into the exact format its chosen model needs. Each model type trains independently: one recipe is one task, one backend and one model.

This sub-project replaces build-order items 2–4 of the parent spec (§13) with a training-first slice. It covers prepare, train, resume, a cheap eval and export for four backends. Playgrounds, the LLM judge, UTMOS, speaker similarity, rating sessions, leaderboards and the voice call stay in later sub-projects.

## 2. Scope

### 2.1 In scope

| Backend | Task | Models (catalog id) | Methods |
|---|---|---|---|
| `unsloth` | llm-sft | `qwen3-4b-instruct-2507`, `gemma-3-4b-it`, `titulm` | `lora`, `qlora`, `full` |
| `unsloth` | asr | `whisper-large-v3-turbo`, `whisper-large-v3-turbo-bn` | `lora` |
| `hf` | asr | `whisper-large-v3-turbo`, `whisper-large-v3-turbo-bn` | `full` |
| `hf` | asr | `wav2vec2-xls-r-300m` | `full` |
| `hf` | asr | `mms-1b-all` | `adapter` |
| `hf` | tts | `indic-parler-tts` | `full` |
| `nemo` | asr | `indicconformer`, `bangla-conformer`, `nemotron-3-5-asr-streaming` | `full` |
| `f5` | tts | `indicf5` | `full` |

Also in scope:
- the shared prepare contract
- `bn_normalize`
- rule metrics
- a shared backend runtime
- two run-detail UI panels
- a VM smoke script

### 2.2 Out of scope

- The Zipformer (icefall/k2), Omnilingual, Piper, Kokoro, XTTS v2, Qwen3-TTS and MMS-TTS backends. Their catalog entries stay browsable but have no trainable backend.
- Serving (`mlops_backend serve`) and playgrounds.
- LLM-as-judge, UTMOS, ECAPA speaker similarity, champion-ASR intelligibility, human rating, leaderboards.
- Voice call.

## 3. Architecture

```
app.core.prepare      manifest reading, PrepSpec, cache dir, atomic build, prepared.json   (pure Python; pyarrow lazy)
app.core.text         bn_normalize                                                         (pure Python)
app.core.metrics      WER/CER, Bangla script share, markdown/emoji rate                    (pure Python)
app.core.backend_kit  phase CLI, Reporter, SIGTERM, resume, OOM/NaN handling               (pure Python)

workers/backends/<name>/
  pyproject.toml, uv.lock          own venv; depends on bangla-voice-backend[prepare] by path
  capabilities.json                tasks, methods, export formats, JSON Schema per task
  mlops_backend/__main__.py        backend_kit.main(handlers)
  mlops_backend/prepare/<family>.py   one converter per model family
  mlops_backend/train.py, eval.py, export.py
  tests/                           converter + contract tests (CPU)
```

- The API and worker still never import trainer code. The worker keeps writing `config.json` and starting `python -m mlops_backend <phase> --run-dir`. `RunConfig` does not change.
- The base install of `bangla-voice-backend` stays `pydantic` + `pyyaml`. A new extra, `prepare = ["pyarrow"]`, carries manifest reading. `app.core.prepare` imports pyarrow inside functions. The test in `tests/core/test_import_isolation.py` is extended to import `app.core.prepare`, `app.core.text`, `app.core.metrics` and `app.core.backend_kit` and still assert no heavy module is loaded.
- `app.core.metrics` has no third-party dependency: WER and CER use its own edit distance.
- Each new backend directory is added to `[tool.uv.workspace] exclude` in `backend/pyproject.toml`, so `scripts/setup_venvs.sh` syncs it.

## 4. Prepare contract

### 4.1 API

```python
class PrepSpec(BaseModel):
    family: str      # "chat", "whisper", "ctc", "parler", "nemo-manifest", "f5"
    version: int     # bumped when the converter's output changes
    params: dict     # everything that changes the output: base model, template, sample rate,
                     # max length, vocab hash, normalizer version

def iter_rows(ds: DatasetRef, split: str) -> Iterator[dict]
    # manifest rows of one split; audio paths made absolute under ds.root

def prepared_dir(cfg: RunConfig, ds: DatasetRef, spec: PrepSpec) -> Path
    # <cfg.prepared_root>/<ds.hash>/<sha256(canonical json of spec)[:16]>/

def ensure_prepared(cfg, ds, spec, build, *, max_skip_share=0.05) -> PrepInfo
    # build(rows_by_split, out_dir) -> BuildResult(counts, skipped, files)
```

### 4.2 Behaviour

- **Hit**: `prepared.json` exists in the target directory. Return its `PrepInfo` without building.
- **Miss**:
  1. Take an exclusive file lock `<target>.lock`.
  2. Check again for a hit.
  3. Delete stale `<target>.tmp-*` directories.
  4. Build into `<target>.tmp-<pid>`.
  5. Write `prepared.json`.
  6. `os.replace` it to the target.
  7. Release the lock.
- **Crash**: a crash mid-build leaves only a tmp directory. A tmp directory is never read.
- **Skipped rows**: a converter skips rows it cannot convert, such as audio over Whisper's 30 s window, a sample longer than the max token length, or text with characters outside a fixed vocabulary. The reason is recorded for each skipped row.
- **Skip threshold**: if skipped rows divided by input rows is above `max_skip_share`, the run fails and the cache is not stored. Default 5 %, overridable with recipe `hparams.prepare.max_skip_share`.
- **Stored data**: the stored dataset is never modified.

### 4.3 `prepared.json`

```json
{
  "spec": {"family": "whisper", "version": 1, "params": {...}},
  "spec_hash": "3f9c...",
  "dataset_hash": "ab12...",
  "counts": {"train": 9500, "val": 480},
  "skipped": [{"sample_id": "s-0042", "split": "train", "reason": "duration 34.2 s > 30 s"}],
  "files": ["train/", "val/"],
  "built_at": 1795000000.0,
  "build_s": 312.4
}
```

During prepare, the backend writes `<run-dir>/prepared_refs.json`: a list with one entry per prepared directory it used (`role` = `train` or `eval`, `path`, `spec_hash`, `cache_hit`, plus that directory's `prepared.json` content). The worker forwards the spec hashes to MLflow as params. This records "prepared-data hashes" (parent spec §5.5).

### 4.4 Converters

| Family | Backend | Input fields | Output |
|---|---|---|---|
| `chat` | unsloth | `messages`, `tools` | Tokenized HF `datasets` saved to disk. The base model's chat template is applied with the tokenizer: Qwen3, Gemma 3 and Llama 3.2 templates differ. Labels mask everything except assistant turns. Samples over `max_seq_length` are skipped. |
| `whisper` | unsloth, hf | `file`, `text` | 16 kHz mono audio, 80/128-bin log-mel input features (per the model's feature extractor), label ids from the Whisper tokenizer with the Bengali language and transcribe task tokens. Audio over 30 s or labels over 448 tokens are skipped. |
| `ctc` | hf | `file`, `text` | 16 kHz audio. wav2vec2: a character vocabulary built from the train split after `bn_normalize`, written as `vocab.json`. MMS: the checkpoint's `ben` adapter vocabulary is used, and rows with characters outside it are skipped. |
| `parler` | hf | `file`, `text`, `speaker`, `style` | 44.1 kHz audio encoded to DAC codes with the checkpoint's audio encoder. Text prompt and a description string. The description is built from a template with `speaker` and `style` (default: `"{speaker} speaks clearly in a studio-quality recording."`), overridable in the recipe. |
| `nemo-manifest` | nemo | `file`, `text` | 16 kHz mono WAV files and one JSONL manifest per split (`audio_filepath`, `duration`, `text`). The checkpoint's tokenizer is kept. Rows whose text the tokenizer cannot represent (an `<unk>` share above 0) are skipped. |
| `f5` | f5 | `file`, `text`, `speaker` | 24 kHz mono audio, the F5 arrow dataset and `duration.json`. `vocab.txt` is the checkpoint's vocabulary with any new characters appended at the end, never reordered. One reference clip per speaker (the median-length clip) is stored for inference. |

ASR and TTS converters pass text through `bn_normalize`. Its version is part of `PrepSpec.params`, so a normalizer change invalidates cached outputs.

## 5. `bn_normalize`

`app.core.text.bn_normalize(text, *, mode)` with `mode` in `{"asr", "tts"}`. Version constant `BN_NORMALIZE_VERSION`.

Both modes:
- Unicode NFC.
- Remove zero-width characters except ZWJ/ZWNJ inside conjuncts.
- Map Bengali digits to one form.
- Unify dandas (`।`, `|`) and quotes.
- Collapse whitespace.

`asr` mode:
- Lowercase Latin letters.
- Strip punctuation.
- This is the form used for WER/CER and CTC vocabularies.

`tts` mode:
- Keep sentence punctuation.
- Expand numbers to Bangla words (integers, decimals, ordinals).
- Expand currency (`৳`, `টাকা`) and percentages.
- Expand times (`hh:mm`) and dates (`dd/mm/yyyy`).

Table-driven tests cover each rule.

## 6. Backend runtime (`app.core.backend_kit`)

- `main(handlers: dict[Phase, Callable[[RunDir, RunConfig, Reporter], int]])`:
  - parses `<phase> --run-dir`
  - reads `config.json`
  - runs the handler and exits with its code
  - an unhandled exception writes `status.message` = exception type and first line, prints the traceback to stderr and exits 1
- `Reporter`:
  - `status(phase, step, total_steps, message)` computes ETA from step rate and writes at most once per second, except on phase change.
  - `metric(step, split, name, value)` appends to `metrics.jsonl`.
- SIGTERM sets a flag. Trainer adapters check it each step, save a checkpoint and exit 143.
- `latest_checkpoint(rd)` returns the newest complete checkpoint under `checkpoints/`. A checkpoint is complete when it has a `.complete` marker written after save. If `cfg.resume` is set, training resumes from it.
- `torch.cuda.OutOfMemoryError` is caught. The handler prints `RuntimeError: CUDA out of memory` (the line the worker already parses), then exits 1.
- NaN or infinite loss on 3 consecutive logged steps stops training with message `loss is NaN; lower learning_rate`.
- `ContractCallback` is a `transformers.TrainerCallback` that maps `on_log`, `on_step_end` and `on_save` to the Reporter, the SIGTERM check and `.complete` markers. It lives in `app.core.backend_kit.hf`, a submodule that imports transformers and is imported only by the HF-based backends (unsloth, hf, f5). The isolation test does not import that submodule. NeMo has an equivalent Lightning callback in `app.core.backend_kit.lightning`, under the same rule.

## 7. Backends

### 7.1 Training, resume, export

| Backend | Train | Resume from | Export formats |
|---|---|---|---|
| `unsloth` | LLM: `FastLanguageModel` with TRL `SFTTrainer`; LoRA, QLoRA (4-bit) or full. Whisper: `FastModel` with LoRA and `Seq2SeqTrainer`. | HF Trainer checkpoint dir | LLM: `merged-safetensors`, `gguf` (quantizations `q4_k_m`, `q8_0`, `f16`, chosen in the recipe). Whisper: `merged-safetensors`, `ct2` (via `ct2-transformers-converter`, `float16`). |
| `hf` | Whisper full: `Seq2SeqTrainer`. wav2vec2: `Trainer` with CTC head on the new vocabulary. MMS: only the `ben` adapter weights train. Parler: the upstream training loop ported into a `Trainer`-style loop with the same callback. | HF Trainer checkpoint dir | `safetensors`; Whisper also `ct2`; MMS adapter also `adapter-safetensors`. |
| `nemo` | `ASRModel.from_pretrained`/`restore_from`, Lightning trainer, `bf16-mixed`. CTC, RNNT and hybrid CTC-RNNT decoders. Nemotron keeps its cache-aware streaming config. | Lightning `.ckpt` | `nemo` |
| `f5` | `f5_tts` `Trainer` from the IndicF5 checkpoint with frame-based batching. EMA kept. | `model_last.pt` | `f5-ckpt` (EMA weights, `vocab.txt`, per-speaker reference clip + transcript) |

### 7.2 Export manifest

Each format is exported on its own. If one fails, the error goes to the log and the others continue. `export/manifest.json` lists only artifacts that exist. The phase exits 0 if at least one format succeeded. The run message then lists the failed formats.

### 7.3 Capabilities and recipes

- Each `capabilities.json` lists its tasks, methods, export formats and a JSON Schema for `method` and `hparams`. `serve_formats` stays empty in this sub-project.
- Schemas restrict `base_model` to the supported model ids.
- Schemas bound hyperparameters to what fits on a 32 GB RTX 5090. For example, `full` is offered for the 4B LLMs only with gradient checkpointing and 8-bit Adam enforced.
- The existing recipe form and validation use these schemas unchanged.
- Catalog `recipe_template` blocks are checked by a test: each must validate against its backend's schema.

### 7.4 Environments

- Each backend pins `torch==2.7.1+cu128` (sm_120) from the PyTorch cu128 index. Lockfiles are resolved for Linux x86_64.
- A backend that cannot use that torch gets a different one in its own venv only. The change is recorded in its `pyproject.toml` comment and in this spec's open items.
- Gated weights (IndicF5, Gemma 3, IndicConformer) need `HF_TOKEN`. It is added to `.env.example` and passed through the worker's subprocess environment. Prepare checks access first and fails with `no access to <repo>: accept its terms on Hugging Face and set HF_TOKEN`.
- Weights cache to `$DATA_ROOT/models/hf` (`HF_HOME`).

## 8. Eval phase

### 8.1 Data and outputs

- Eval data: `datasets.eval` (split `test`, or every split for eval-only datasets) when the recipe sets it. Otherwise the train dataset's `val` split.
- Outputs:
  - `eval/results.json`: metrics
  - `eval/samples.jsonl`: one line per sample
  - `eval/audio/*.wav`: TTS only

### 8.2 Metrics per task

| Task | Metrics | Sample line |
|---|---|---|
| llm-sft | `eval_loss`, `perplexity`. Then greedy generation (max 512 new tokens) for up to 200 prompts, scored with `bangla_script_share`, `markdown_rate`, `emoji_rate`, `avg_reply_chars`. | `prompt`, `reference`, `reply` |
| asr | `wer`, `cer` after `bn_normalize(mode="asr")`, overall and `wer/<slice>=<value>` for `region`, `channel` and `noise` when present; `rtf` | `sample_id`, `reference`, `hypothesis`, `wer` |
| tts | `eval_loss`; `rtf` over up to 20 synthesized eval sentences | `sample_id`, `text`, `audio` |

### 8.3 Baseline

- Before training, the eval phase scores the untouched base model on the same data. Its metrics are stored with a `base/` prefix.
- Recipe `hparams.eval.baseline: false` skips this.
- The baseline is cached per (base model, eval dataset hash, eval spec) under `prepared_root/_baselines/`, so it is not recomputed for every run.

## 9. API and UI

- `GET /runs/{id}/eval/samples?offset=&limit=` returns paged `samples.jsonl` lines.
- `GET /runs/{id}/eval/audio/{name}` serves a WAV from the run's `eval/audio/`. The resolved path must stay inside that directory, otherwise 404.
- `GET /runs/{id}/prepared` returns the run's `prepared_refs.json`, with counts and skipped rows.
- Run detail page:
  - **Prepared data** card: family, spec hash, cache hit or miss, counts per split, skipped rows (first 50, with reasons).
  - **Eval samples** panel:
    - LLM: a table of prompt, reference and reply.
    - ASR: reference/hypothesis word diff, sortable by WER.
    - TTS: an audio player per sentence.
  - Metrics with a `base/` twin show the delta.

## 10. Error handling

| Where | Condition | Result |
|---|---|---|
| prepare | no access to gated weights | run `failed`, message names the repo and `HF_TOKEN` |
| prepare | skipped share above threshold | run `failed`, message gives the share and the top 3 skip reasons; nothing cached |
| prepare | undecodable audio | row skipped with reason (ingest already blocks these; this is a guard) |
| train | CUDA OOM | existing OOM hint in the UI; checkpoint kept |
| train | 3 NaN/inf losses | run `failed`, `loss is NaN; lower learning_rate` |
| train | cancel (SIGTERM) | checkpoint saved, exit 143, resumable |
| export | one format fails | other formats still exported; run `done` with a warning message |
| recipe | method or hparams outside the backend schema | rejected at validation, before any GPU time |

## 11. Testing

### 11.1 Local, CPU only

These run with no GPU, no network and no weights, in the backend workspace (`cd backend && uv run pytest`):
- `app.core.prepare`:
  - cache hit and miss
  - lock contention between two processes
  - crash mid-build leaves no target
  - stale tmp cleanup
  - skip threshold
- `bn_normalize`: table tests per rule and mode.
- `app.core.metrics`: WER/CER against known values; script share, markdown and emoji detection.
- `backend_kit`:
  - phase dispatch
  - exception to status message
  - SIGTERM flag
  - `latest_checkpoint` ignores incomplete checkpoints
  - OOM line
  - NaN stop
- API: the three new endpoints, including path traversal on the audio endpoint.
- Import isolation: extended per §3.
- Catalog `recipe_template` blocks validate against backend schemas.

### 11.2 Per-backend tests

These run in each backend's own venv (`uv run --project backend/workers/backends/<name> pytest`):
- Every converter on a tiny fixture dataset under `tests/fixtures`. Fixtures include tiny tokenizer and vocabulary files, so no model download is needed. Converters that need a full model component (DAC encoder, Whisper feature extractor) are marked `needs_weights` and run only on the VM.
- Contract behaviour of each phase with the trainer replaced by a stub:
  - status
  - metrics
  - checkpoints
  - export manifest with partial failure
- These run on Linux. The plan does not promise they run on Windows, because the backend venvs target Linux.

### 11.3 VM smoke

`scripts/smoke_backends.sh [backend...]` covers each backend in turn:
1. Uploads a tiny bundled dataset through the API.
2. Creates a tiny recipe.
3. Runs about 20 steps, then eval, then export.
4. Checks the run is `done`, `prepared_refs.json` exists, `eval/results.json` has the task's metrics and each export artifact exists.

The user runs it on the VM.

## 12. Delivery order

Each step ends with its tests green and is usable on its own.

1. Shared core:
   - the `prepare` extra
   - `app.core.prepare`, `app.core.text`, `app.core.metrics`, `app.core.backend_kit`
   - the extended isolation test
   - the three API endpoints and the two UI panels
2. `unsloth`:
   1. VM install and import check (torch, unsloth, sm_120).
   2. `chat` converter, LLM train/eval/export.
   3. Whisper LoRA.
3. `hf`:
   1. VM install check.
   2. `whisper` (shared with unsloth), `ctc` and `parler` converters, then their training.
4. `nemo`:
   1. VM install check.
   2. `nemo-manifest` converter and training for the three models.
5. `f5`:
   1. VM install check.
   2. `f5` converter and training.
6. Smoke script, README section, `.env.example` (`HF_TOKEN`, `HF_HOME`).

If a backend's VM install check fails (Unsloth or NeMo on sm_120 are the known risks), work on that backend stops. The failure is reported with the exact error, and the next backend proceeds.

## 13. Open items

- Unsloth and NeMo on sm_120 with `torch==2.7.1+cu128` (parent spec §14).
- Whether Parler-TTS training code installs cleanly with the pinned torch and transformers versions.
- Licenses carried over from the parent spec: IndicConformer, BanglaConformer, IndicF5 weights.
- GGUF export needs llama.cpp built on the VM. Unsloth builds it on first use; the build tools (`cmake`, `build-essential`) are added to `scripts/check_vm.sh`.
