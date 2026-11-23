# Bangla Voice MLOps Platform: Design

Date: 2026-11-23
Status: approved in conversation, awaiting written-spec review

## 1. Purpose

Build a standalone platform for fine-tuning, evaluating and trying out the three models behind a Bangla e-commerce voice bot:

- **LLM**: an instruction model fine-tuned to answer like a Bangladeshi e-commerce customer-service assistant.
- **STT**: a speech recognizer fine-tuned on Bangladeshi conversational and telephone Bangla.
- **TTS**: a speech synthesizer fine-tuned on a Bangladeshi voice.

The training pipeline must not be tied to any one model. A new base model is a new recipe file; a new training framework is one new backend adapter.

### 1.1 Users and context

- One operator (the project owner) trains, evaluates and listens. Colleagues only source datasets and hand them over. No accounts, roles or auth.
- Internal R&D now, production later. Production-bound models must use commercially usable base weights and data. Non-commercial checkpoints are allowed only as benchmarks.
- Compute is one Ubuntu 24.04 VM with an RTX 5090 (32 GB, Blackwell `sm_120`), the same VM that runs LocalVoiceAgent. Use `torch==2.7.1+cu128` or newer that supports `sm_120`. Never use the host `nvcc` 12.0.
- Workflow: edit on the Windows PC, push to git, pull on the VM, run on the VM. The UI is opened from the PC through an SSH tunnel.

### 1.2 Success criteria

1. Upload a dataset through the UI, see it validated and versioned.
2. Launch a fine-tuning run from a recipe for any supported (task, backend, base model) combination, watch it live, cancel and resume it.
3. Every run ends with evaluation metrics, exported artifacts and a license badge.
4. Compare models side by side in the playground and on leaderboards, including blind human ratings.
5. Talk to any combination of registered STT + LLM + TTS models in the voice call page, with barge-in, diarization and RAG, and get per-turn latency.
6. Adding a new base model needs no code change when its backend already supports the architecture.

### 1.3 Out of scope (v1)

- Multi-user access, auth, remote or cloud compute, multi-GPU training.
- Automatic hyperparameter search (clone a recipe instead).
- Dataset import from Hugging Face Hub or buckets (UI upload only).
- Production serving and telephony. The voice call page is an evaluation tool.

## 2. Initial model shortlist

Primary candidates are commercially usable. Benchmark-only entries are marked.

| Task | Primary | Comparison | Benchmark only (non-commercial) |
|---|---|---|---|
| LLM | Qwen3-4B-Instruct-2507 (Apache 2.0) | Gemma 3 4B-it (Gemma Terms), TituLM | none |
| STT | Streaming FastConformer from BanglaConformer or IndicConformer (verify licenses) | Whisper-large-v3-turbo (MIT), wav2vec2 XLS-R 300M (Apache 2.0) | MMS-1B-all |
| TTS | IndicF5 (MIT, verify) | Indic Parler-TTS (Apache 2.0) | `ehzawad/indicf5-bangla-tts`, F5-TTS base weights, MMS-TTS |

Check licenses on each model card before a model is promoted for production. Check Hugging Face for Bangla releases newer than mid-2026.

## 3. Architecture

```
web/        Next.js (App Router) + shadcn/ui
api/        FastAPI + async SQLAlchemy + Postgres + Alembic
worker/     database-backed job queue, single GPU lock, launches backend subprocesses
backends/   unsloth/ hf/ nemo/ f5/   (each its own uv venv + backend adapter)
voice/      voice runtime copied from LocalVoiceAgent + vendored RealtimeSTT/RealtimeTTS
shared/     bn_normalize, schemas, run-dir contract helpers (stdlib + pydantic only)
storage:    $DATA_ROOT/data  $DATA_ROOT/runs  $DATA_ROOT/models
mlflow      metrics and params; artifact links point into runs/
```

### 3.1 Three layers of model independence

1. **Task type**: `llm-sft`, `asr`, `tts`. Each task fixes a dataset schema, a recipe schema for common fields and a set of evaluation metrics.
2. **Recipe**: a YAML file naming the base model, backend, method and hyperparameters.
3. **Backend**: one training framework behind the run-directory contract (§5.2).

The API and worker never import trainer code. They only write `config.json`, start a subprocess and read the files the backend writes.

### 3.2 Processes and deployment

- docker-compose runs Postgres, MLflow, the API and the web UI. The job queue lives in Postgres: one worker claims jobs with a conditional state update, so no Redis is needed.
- The worker, backends and voice runtime run natively on the VM, each in its own uv venv. This matches LocalVoiceAgent, where uv hardlinks one torch copy into every venv.
- The worker processes one GPU job at a time. CPU jobs (dataset ingest, CPU-only evaluation steps) run in a separate queue and do not take the GPU lock.

### 3.3 GPU lock

- One lock for the GPU. A training or GPU evaluation job holds it for its whole run.
- Playground and voice call models hold it while loaded. They unload on request or after 15 minutes idle.
- Starting a training job while models are loaded shows a prompt to unload them. Training is never preempted automatically.
- The dashboard shows the lock holder, VRAM, utilization and temperature (`nvidia-smi` / NVML).

## 4. Data

### 4.1 Upload and ingest

- Resumable chunked upload (tus protocol) into `data/incoming/`. Audio datasets can be tens of GB.
- A CPU ingest job validates the upload, writes a Parquet manifest and registers a dataset version.
- Provenance fields are required at upload: `source`, `license`, `consent`, `commercial_ok` (bool), free-text `notes`.

### 4.2 Formats

| Task | Upload | Required fields | Optional fields |
|---|---|---|---|
| `llm-sft` | `.jsonl` | `messages[]` with roles `system`, `user`, `assistant`, `tool` | `tools`, `tags` |
| `tts` | `.zip` with `metadata.csv` + `wavs/` | `file`, `text`, `speaker` | `style` |
| `asr` | `.zip` with `metadata.csv` + `audio/` | `file`, `text` | `region`, `channel` (`studio`/`phone`), `noise`, `speaker` |

### 4.3 Validation

Errors block registration. Warnings are shown but do not block.

- **llm-sft errors**: invalid JSON, bad role order, empty assistant turn. **Warnings**: Bangla script share below 50 % in assistant turns, markdown or emoji in assistant turns (the output is spoken), duplicates, samples over 12,000 characters (about 4,096 tokens; no tokenizer is loaded at ingest).
- **audio errors**: undecodable file, missing transcript, transcript without audio. **Warnings**: duration outside 0.5–30 s, clipping, silence share above 50 %, loudness outside −30 to −14 LUFS, mixed sample rates.

The UI shows the report and a sample browser: play audio, read text, filter by warning.

### 4.4 Versioning and splits

- A dataset version is immutable and addressed by the hash of its manifest.
- Editing a transcript in the UI creates a new version.
- Splits are deterministic by hash of sample ID. Ratios are set per version (default 90/5/5).
- A dataset can be flagged `eval-only`. Eval-only datasets are refused by recipe validation as training data.

### 4.5 Backend preparation

Resampling, feature extraction, NeMo manifests and chat-template rendering happen in the backend's `prepare` step. The output is cached under `data/prepared/<dataset-hash>/<prep-spec-hash>/`. The stored dataset never changes.

### 4.6 Licensing

- A run's `commercial_ok` is the AND of its base model's flag (from the recipe's model metadata) and every training dataset's flag.
- The UI shows the result as a badge on runs and model versions.

### 4.7 Bangla text normalizer

`shared/bn_normalize` converts numerals, ৳ amounts, phone numbers, dates and times into spoken Bangla. It also handles common English tokens in Bangla text. It is used in TTS data preparation, TTS inference, and ASR scoring (both texts normalized before WER/CER). It has its own test table of input/expected pairs.

## 5. Training

### 5.1 Recipe

```yaml
name: qwen3-4b-ecom-qlora
task: llm-sft
backend: unsloth
base_model: Qwen/Qwen3-4B-Instruct-2507
base_model_meta: {license: apache-2.0, commercial_ok: true}
datasets:
  train: ecom-instruct@v3
  eval: ecom-eval@v1
method: {type: qlora, r: 16, alpha: 32, dropout: 0.0}
hparams: {lr: 2e-4, epochs: 3, batch_size: 8, grad_accum: 2, max_seq_len: 2048, seed: 42}
export: [merged, "gguf:q4_k_m"]
```

- The task schema validates common fields. The backend validates its own fields through a JSON Schema in its `capabilities.json`.
- The UI builds a form from these schemas and also offers a raw YAML editor.
- The database is the source of truth for recipes. Each save also writes a read-only copy to `recipes/<name>.yaml` so recipes can be reviewed in git. A recipe used by a run is frozen; editing it creates a new recipe version.

### 5.2 Run-directory contract

```
runs/<run-id>/
  config.json     resolved recipe, dataset paths, prepared-data paths, seed   (platform writes)
  status.json     {phase, step, total_steps, eta_s, message}                  (backend writes)
  metrics.jsonl   {step, split, name, value, ts} per line                    (backend writes)
  logs/           stdout.log, stderr.log                                     (worker captures)
  checkpoints/    backend-specific; latest is resumable
  eval/           results.json + samples/ (text, audio)
  export/         manifest.json + artifacts
```

Backend entry point, run inside the backend's venv:

```
python -m mlops_backend <prepare|train|eval|export|serve> --run-dir runs/<run-id>
```

Each backend ships `capabilities.json` listing its tasks, methods, export formats, serve formats and the JSON Schema for its recipe fields. The UI only offers combinations listed there.

`export/manifest.json` lists each artifact with `format` (e.g. `gguf`, `ct2`, `nemo`, `f5-ckpt`, `safetensors`), `path`, `size_bytes` and `serve_with` (the runtime that can load it).

### 5.3 Lifecycle

`queued -> preparing -> training -> evaluating -> exporting -> done | failed | cancelled`

- Cancel sends SIGTERM. The backend saves a checkpoint and exits within 60 s, else SIGKILL.
- `failed` and `cancelled` runs can resume from their latest checkpoint.
- The worker tails `status.json` and `metrics.jsonl`, forwards them to MLflow and to the UI over SSE.
- On startup the worker marks runs left in a running phase as `failed`, so they can be resumed.

### 5.4 Backends

| Backend | Tasks and methods | Export formats |
|---|---|---|
| `unsloth` | llm-sft (Qwen3, Gemma 3, Llama; LoRA/QLoRA/full), asr (Whisper LoRA) | merged safetensors, GGUF; Whisper to CTranslate2 |
| `hf` | asr (Whisper full fine-tune, wav2vec2 CTC), tts (Parler-TTS) | safetensors, CTranslate2 |
| `nemo` | asr (FastConformer CTC/RNNT, cache-aware streaming) | `.nemo` |
| `f5` | tts (IndicF5, F5-TTS fine-tune) | checkpoint + vocab |

A backend that needs a different torch version than the pinned one is recorded and handled in its own venv; it does not change other venvs.

### 5.5 Reproducibility

Each run records: git commit SHA, hash of the backend's `uv.lock`, seed, dataset version hashes, prepared-data hashes and the resolved config.

### 5.6 Model registry

- Promoting a `done` run creates a model version: task, base model, recipe, evaluation metrics, exports, `commercial_ok`.
- Each task has at most one `champion`. The voice call page and TTS intelligibility scoring use champions by default.

## 6. Evaluation

### 6.1 When it runs

- As the `evaluating` phase of every run, on the recipe's eval datasets.
- As a standalone eval job for any model version or untrained base model against any `eval-only` dataset. This gives zero-shot baselines.

### 6.2 Metrics

| Task | Automatic | Human |
|---|---|---|
| llm-sft | LLM-as-judge rubric (helpfulness, Bangla fluency, politeness, brevity, speakability, 1–5 each); markdown/emoji rate; Bangla script share; tokens per word; TTFT; tokens/s; tool-call accuracy when the eval set has tool calls | Blind A/B preference |
| asr | WER and CER after `bn_normalize`, overall and per slice (`region`, `channel`, `noise`); real-time factor; latency | Transcript diff view |
| tts | Intelligibility: champion ASR transcribes the audio, CER vs input; speaker similarity (ECAPA embedding cosine); predicted MOS (UTMOS); time to first audio; real-time factor | Blind MOS 1–5, blind A/B |

- The judge is pluggable: a local served model, or a Claude/OpenAI API key set in settings. The settings page states that API judges send eval samples to that provider.
- Leaderboards are per (task, eval dataset version). Any two runs can be compared metric by metric.

### 6.3 Human rating

- A rating session draws samples from chosen models, hides model identities and shuffles order.
- Ratings are stored per sample and aggregated into MOS (mean and 95 % CI) or win rate.

## 7. Playground

- **LLM**: chat with up to 3 models side by side; system prompt; temperature, top-p, max tokens; live TTFT and tokens/s.
- **TTS**: enter text, see the `bn_normalize` output, hear up to 3 models.
- **STT**: record in the browser or upload audio; transcripts from up to 3 models, diffed against an optional reference.
- **Serving**: the API starts `mlops_backend serve` for the chosen export in its backend venv, as a local HTTP server, and proxies to it. LLM GGUF exports are served by llama-server (Docker image, as in LocalVoiceAgent). Loaded models hold the GPU lock (§3.3).

## 8. Voice call

### 8.1 Source

Copy from LocalVoiceAgent branch `feat/call-UI-integration` (commit `74b0e93` or later), which already contains v2 model switching, v3 barge-in, v3 RAG and v4 diarization. LocalVoiceAgent itself is not modified. Record the source commit in `voice/VENDORED.md`. Integration context is in `docs/context/localvoiceagent.md`.

Copied:
- `server/voicebot/`: session, timing, speech filter, audio I/O (browser mode), barge-in, diarizer, RAG, latency report, and their tests.
- `RealtimeSTT/` and `RealtimeTTS/` vendored copies, unedited, with their `VENDORED.md` workarounds.
- The `/call` page UI, ported from React/Vite to the Next.js app.

### 8.2 Adapters to the model registry

- **LLM**: GGUF export served by llama-server through the existing `LLMBackend` protocol.
- **STT**: Whisper exports in CTranslate2 run on faster-whisper. NeMo exports need a new engine class in `voice/` that subclasses the RealtimeSTT engine base. Vendored code stays unedited.
- **TTS**: F5/IndicF5 and Parler exports need new engine classes in `voice/` that subclass the RealtimeTTS engine base.
- **RAG**: the embedder must handle Bangla. Default to a multilingual embedder such as `BAAI/bge-m3`; measure Bangla retrieval before relying on it. Knowledge sources are managed on the Knowledge page.
- **Diarization**: Nemotron streaming diarizer as in LocalVoiceAgent. Check its license before production.

### 8.3 Behavior

- The call page picks one STT, LLM and TTS model version. Defaults are the champions.
- Barge-in, diarization and RAG are toggles.
- Every turn's timing spans (`silence_wait_ms`, `stt_ms`, `llm_ttft_ms`, `filter_ms`, `tts_first_audio_ms`, `playback_ms`, `perceived_ms`, `total_ms`) are stored with the model combination, giving a latency leaderboard.

## 9. Frontend

Next.js App Router, shadcn/ui, Tailwind, dark mode. Live data through SSE. Opened only through the SSH tunnel; no auth.

| Page | Content |
|---|---|
| Dashboard | GPU status and lock holder, job queue, recent runs, champions per task, disk free |
| Datasets | Upload with progress and resume, validation report, sample browser, transcript edit (new version), provenance fields |
| Recipes | Schema form and YAML editor, capability-filtered choices, clone |
| Runs | List; detail page with live metric charts, streamed logs, phase timeline, cancel/resume, eval samples, exports, promote |
| Models | Registry, versions, license badge, champion, download exports |
| Evaluate | Launch eval jobs, leaderboards, run comparison, rating sessions |
| Playground | LLM / TTS / STT tabs |
| Voice Call | Model picker, feature toggles, call UI, per-turn latency breakdown |
| Knowledge | RAG source upload and ingest status |
| Settings | Judge provider and key, data root, idle-unload timeout, disk thresholds |

## 10. Error handling

- Backend crash: run becomes `failed`; the UI shows the last 200 log lines; checkpoints are kept for resume.
- CUDA out of memory: detected from logs; the UI suggests a lower batch size or gradient checkpointing.
- Worker restart: runs left in a running phase become `failed` (resumable).
- Upload: validation happens before any GPU time is used.
- Disk: warn below 20 GB free on the data root; refuse new runs and uploads below 10 GB.
- Serve process crash: model marked unloaded, GPU lock released, error shown in the playground.

## 11. Testing

- API, worker, schemas, `bn_normalize` and the run-dir contract are tested with pytest and a **fake backend**: a small script that writes the contract files without training. Tests need no GPU, network, Docker or model weights.
- Each real backend has a GPU smoke test on the VM: about 20 steps on a tiny dataset, then export and serve.
- Voice runtime tests come over with the copied code and keep passing.
- Web: vitest, typecheck and build.

## 12. Repository layout

```
api/            FastAPI app, models, routers, Alembic
worker/         queue consumer, GPU lock, subprocess runner
shared/         bn_normalize, task schemas, run-dir helpers
backends/
  unsloth/      pyproject.toml, uv.lock, mlops_backend/, capabilities.json
  hf/
  nemo/
  f5/
  fake/         test backend
voice/          copied voicebot runtime + new engine adapters
vendor/         RealtimeSTT/, RealtimeTTS/
web/            Next.js app
recipes/        exported recipe YAML
docker-compose.yml, .env.example, scripts/ (setup_venvs.sh, check_vm.sh, dev.sh)
docs/           context/, superpowers/specs/, superpowers/plans/
```

## 13. Build order (sub-projects)

Each sub-project gets its own implementation plan.

1. **Core platform**: compose services, API, database, worker, GPU lock, run-dir contract, fake backend, dataset upload/validation/versioning, recipes, runs, registry, minimal UI for these pages.
2. **LLM track**: `unsloth` backend for llm-sft, GGUF export, LLM eval (judge + rules), LLM playground.
3. **ASR track**: `hf` and `nemo` backends, `bn_normalize`, ASR eval, STT playground.
4. **TTS track**: `f5` backend (and Parler on `hf`), TTS eval and rating sessions, TTS playground.
5. **Voice call**: copy the voice runtime, registry adapters, new engine classes, RAG and diarization, call page.

Step 2 can start before any audio data exists.

## 14. Open items to verify during implementation

- Licenses of BanglaConformer, IndicConformer, IndicF5 weights, UTMOS, Nemotron diarizer.
- Unsloth and NeMo installs on `sm_120` with the pinned torch.
- Bangla retrieval quality of the chosen RAG embedder.
- Tokens-per-word of each LLM tokenizer on the project's Bangla data.
