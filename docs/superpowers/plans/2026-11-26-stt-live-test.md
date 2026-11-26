# STT Live Test Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A built-in `/test/stt` page that loads a Whisper or Qwen3-ASR model, transcribes the browser mic live with LocalVoiceAgent's turn-end rule and per-turn latency, has an English/Bangla STT language switch, and a file mode with WER/CER.

**Architecture:** A new native FastAPI service (`backend/voice/`, package `voice_service`, own uv venv) wraps a RealtimeSTT `AudioToTextRecorder`, takes the shared GPU lock as `voice` (with a heartbeat the worker reclaims), and talks to the browser over REST + one WebSocket. Qwen3-ASR runs through a custom RealtimeSTT engine registered in every process of the voice venv by a `.pth` hook, using a prompt module shared with the qwen-asr training backend. The Next.js page talks to the voice service directly (`NEXT_PUBLIC_VOICE_URL`) and to the API for lock/queue state.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy async, Alembic, RealtimeSTT (LocalVoiceAgent pin), faster-whisper/CTranslate2, transformers 5.24.1, soxr, torch 2.7.1+cu128; Next.js App Router, shadcn/ui, vitest.

**Spec:** `docs/superpowers/specs/2026-11-26-stt-live-test-design.md`

## Global Constraints

- Python `>=3.12,<3.13`; `torch==2.7.1` from the `pytorch-cu128` index; `transformers==5.24.1`.
- RealtimeSTT: `RealtimeSTT[faster-whisper] @ git+https://github.com/kawsarmahmudsagor/LocalVoiceAgent@fd6bb52258a3154a20d75249ee803dcd99dc3df7#subdirectory=RealtimeSTT`. Never edit RealtimeSTT source; never install the `qwen_asr` pip package.
- Audio reaches the recorder only as continuous 16 kHz mono PCM16, `feed_audio(..., original_sample_rate=16000)`; any other rate is resampled with a per-connection `soxr.ResampleStream`.
- Turn end = `post_speech_silence_duration`, default `0.6` s (`STT_SILENCE_SEC`); beam default `5`.
- Span names exactly: `silence_wait_ms`, `handoff_ms`, `stt_ms`, `speech_end_to_text_ms`.
- Language codes `bn` (default) and `en`; Qwen prefix names `Bengali` / `English`.
- GPU lock: `holder_kind="voice"`; heartbeat every `10` s; stale after `60` s; idle unload after `15` min without a finished turn or file request (audio frames do not count); `POST /voice/load` refused with 409 while any run is `queued`.
- CT2 cache: `$DATA_ROOT/models/ct2-cache/<repo with "/" replaced by "--">-<compute_type>/`.
- Turn log: `$DATA_ROOT/logs/voice-turns.jsonl`.
- `VOICE_PORT` default `8010`, bind `127.0.0.1`; frontend `NEXT_PUBLIC_VOICE_URL` default `http://localhost:8010`.
- Tests run without GPU, network, Docker or model weights, except tests marked `heavy`.
- `app.core` must keep importing without server deps (`backend/tests/core/test_import_isolation.py`).

## Review Focus

- WebSocket closed mid-turn (tab closed, network drop): the listen thread and resampler reset; the next connection works; no lock leak. Test in Task 8.
- Load pressed while a model is loading or already loaded: second load replaces the first cleanly (shutdown old recorder, keep one lock holder), never two recorders. Test in Task 8.
- Language switched during an utterance: that turn uses the language at transcription start; the turn row records it. Test in Task 7.
- Silence-only or very short audio (no speech, cough): no turn row, no crash, idle clock unchanged. Test in Task 7.
- Voice service unreachable or restarting while the page is open: banner with start command, controls disabled, recovers when it returns. Test in Task 11.

---

## File map

```
backend/app/core/backend_kit/qwen_asr_prompt.py   shared Qwen prompt (moved from qwen-asr backend), language-aware
backend/workers/backends/qwen-asr/mlops_backend/prompt.py   re-export shim
backend/alembic/versions/0005_gpu_lock_heartbeat.py
backend/app/models.py                              GpuLock.heartbeat_at
backend/workers/lock.py                            heartbeat(), reclaim_stale()
backend/workers/__main__.py                        gpu_tick reclaims stale voice lock
backend/app/routers/system.py                      lock.heartbeat_at in /api/system
backend/voice/pyproject.toml, voice_engines.pth.txt
backend/voice/voice_service/
  settings.py        VoiceSettings
  app.py             create_app(settings, deps) + routes
  engine_hook.py     registers "bangla_qwen3_asr" in RealtimeSTT's engine table (imported by .pth)
  qwen_engine.py     QwenAsrEngine(BaseTranscriptionEngine)
  models.py          ModelChoice, list_models(), resolve_ct2()
  gpu.py             VoiceLock (acquire/heartbeat/release/startup recovery/queued check)
  live.py            Turn, TurnTimer, StreamResampler, LiveSession
  filemode.py        FileTranscriber
backend/voice/tests/...
scripts/setup_voice_venv.sh, run-vm.sh, .env.example, docker-compose.yml, README.md
frontend/src/lib/voice.ts, voice-types.ts, pcm16.ts, turns.ts (+ tests)
frontend/public/pcm16-worklet.js
frontend/src/app/test/stt/page.tsx, frontend/src/components/stt/*.tsx
frontend/src/lib/nav.ts, app/registry/[id]/page.tsx, app/runs/page.tsx, app/runs/[id]/page.tsx
```

---

### Task 1: Shared, language-aware Qwen3-ASR prompt module

**Files:**
- Create: `backend/app/core/backend_kit/qwen_asr_prompt.py` (contents of `backend/workers/backends/qwen-asr/mlops_backend/prompt.py`, plus language parameters)
- Modify: `backend/workers/backends/qwen-asr/mlops_backend/prompt.py` → `from app.core.backend_kit.qwen_asr_prompt import *` plus explicit re-exports of every public name and `_default_loader`, `_like`
- Test: `backend/tests/core/test_qwen_asr_prompt.py`; existing `backend/workers/backends/qwen-asr/tests/*` unchanged

**Interfaces:**
- Produces: `LANGUAGE_NAMES: dict[str, str] = {"bn": "Bengali", "en": "English"}`; `prefix_for(language: str = "Bengali") -> str` (`f"language {language}{MARKER}"`); `conversation(audio_ref="audio.wav", language="Bengali")`; `prompt_text(processor, language="Bengali") -> str`; unchanged `MARKER`, `PREFIX` (= `prefix_for("Bengali")`), `SAMPLE_RATE`, `split_transcript`, `max_new_tokens`, `generation_inputs`, `QwenAsrCollator`, `mask_labels`, `training_text`.

- [ ] **Step 1: Write failing tests**

```python
from app.core.backend_kit.qwen_asr_prompt import LANGUAGE_NAMES, PREFIX, prefix_for, prompt_text, split_transcript

class FakeProc:
    def apply_chat_template(self, conv, tokenize, continue_final_message):
        return "<sys>...<audio>" + conv[-1]["content"][0]["text"]

def test_prefix_default_is_bengali():
    assert PREFIX == "language Bengali<asr_text>" == prefix_for()

def test_prompt_text_uses_language():
    assert prompt_text(FakeProc(), language="English").endswith("language English<asr_text>")

def test_language_names():
    assert LANGUAGE_NAMES == {"bn": "Bengali", "en": "English"}

def test_split_transcript_after_marker():
    assert split_transcript("language Bengali<asr_text> আমার অর্ডার ") == "আমার অর্ডার"
```

Add `"app.core.backend_kit.qwen_asr_prompt"` to the import list in `test_import_isolation.py`.

- [ ] **Step 2:** `cd backend && uv run pytest tests/core/test_qwen_asr_prompt.py tests/core/test_import_isolation.py -v` → FAIL (module missing).
- [ ] **Step 3:** Move the module, add the language parameters, keep `prompt_text`'s endswith check against `prefix_for(language)`. Write the shim.
- [ ] **Step 4:** Run Step 2 → PASS; `bash scripts/test_backend.sh qwen-asr` → PASS (light tests).
- [ ] **Step 5:** Commit `refactor(qwen-asr): share the prompt module via backend_kit; language-aware prefix`.

---

### Task 2: GPU lock heartbeat and stale-voice reclaim

**Files:**
- Create: `backend/alembic/versions/0005_gpu_lock_heartbeat.py` (revision `"0005"`, down `"0004"`, `batch_alter_table("gpu_lock")` add nullable `heartbeat_at` `sa.DateTime(timezone=True)`)
- Modify: `backend/app/models.py` (`GpuLock.heartbeat_at = _ts(nullable=True)`), `backend/workers/lock.py`, `backend/workers/__main__.py`, `backend/app/routers/system.py`, `frontend/src/lib/types.ts` (`lock.heartbeat_at: string | null`)
- Test: `backend/tests/workers/test_lock.py`, `backend/tests/workers/test_main.py`, `backend/tests/app/test_system_api.py`

**Interfaces:**
- Produces: `acquire()` also sets `heartbeat_at=utcnow()`; `release()` clears it; `heartbeat(session, kind: str, holder_id: str) -> bool` (updates only if that holder holds it); `reclaim_stale(session, kind: str, max_age_s: float, now=None) -> str | None` (releases a lock of `kind` whose `heartbeat_at` is null or older than `max_age_s`, returns the released holder id); constant `VOICE_STALE_S = 60` in `workers/lock.py`.
- `gpu_tick` calls `reclaim_stale(s, "voice", VOICE_STALE_S)` before `claim_next_run`, logging a warning when it releases.
- `/api/system` → `lock.heartbeat_at` (ISO or null).

- [ ] **Step 1: Failing tests**

```python
async def test_heartbeat_only_for_holder(session):
    await acquire(session, "voice", "v-1")
    assert await heartbeat(session, "voice", "v-1")
    assert not await heartbeat(session, "voice", "v-2")

async def test_reclaim_stale_voice(session):
    await acquire(session, "voice", "v-1")
    later = utcnow() + dt.timedelta(seconds=61)
    assert await reclaim_stale(session, "voice", 60, now=later) == "v-1"
    assert (await current(session)).holder_kind is None

async def test_reclaim_ignores_fresh_and_runs(session):
    await acquire(session, "run", "r-1")
    assert await reclaim_stale(session, "voice", 60, now=utcnow() + dt.timedelta(hours=1)) is None
    assert (await current(session)).holder_id == "r-1"
```

`test_main.py`: a stale voice lock plus a queued run → one `gpu_tick` claims the run. `test_system_api.py`: response has `lock.heartbeat_at`.

- [ ] **Step 2:** `cd backend && uv run pytest tests/workers/test_lock.py tests/workers/test_main.py tests/app/test_system_api.py -v` → FAIL.
- [ ] **Step 3:** Implement migration, model column, lock functions, tick call, system field, TS type.
- [ ] **Step 4:** Step 2 → PASS; `uv run alembic -c alembic.ini upgrade head` against a temp sqlite URL succeeds; full `uv run pytest` green.
- [ ] **Step 5:** Commit `feat(lock): heartbeat column; worker reclaims stale voice locks`.

---

### Task 3: Voice service skeleton and venv

**Files:**
- Create: `backend/voice/pyproject.toml`, `backend/voice/voice_service/__init__.py`, `settings.py`, `app.py`, `backend/voice/tests/conftest.py`, `backend/voice/tests/test_app.py`, `scripts/setup_voice_venv.sh`
- Modify: `backend/pyproject.toml` (`exclude` += `"voice"`)

**Interfaces:**
- `pyproject.toml`: name `bangla-voice-service`; deps `bangla-voice-backend[server,audio]` (path `../`, editable), RealtimeSTT pin, `faster-whisper`, `ctranslate2`, `transformers==5.24.1`, `accelerate`, `librosa`, `jinja2`, `soxr`, `torch==2.7.1`, `httpx`, `uvicorn[standard]`; dev `pytest`, `pytest-asyncio`; `asyncio_mode = "auto"`; markers `heavy`; `addopts = "-m 'not heavy'"`.
- `VoiceSettings(BaseSettings)`: `database_url: str`, `api_url: str = "http://localhost:8000"`, `data_root: Path`, `voice_port: int = 8010`, `web_origin: str = "http://localhost:3001"`, `silence_sec: float = 0.6` (env `STT_SILENCE_SEC`), `idle_unload_s: float = 900`, `heartbeat_s: float = 10`, `device: str = "cuda"`.
- `create_app(settings: VoiceSettings, deps: Deps | None = None) -> FastAPI`; `Deps` is a dataclass of injectable factories filled by later tasks (`recorder_cls`, `api_client`, `lock`, `file_transcriber`); default `None` → real implementations. Route `GET /voice/health` → `{"ok": true}`. CORS allows `web_origin`.
- `scripts/setup_voice_venv.sh`: `uv sync` in `backend/voice`; writes `<site-packages>/voice_engines.pth` containing `import voice_service.engine_hook` (pattern: `cosyvoice_upstream.pth` in `scripts/setup_venvs.sh`); prints the site path.
- Run command (used by Task 10): `cd backend/voice && LD_LIBRARY_PATH=<venv nvidia libs> .venv/bin/uvicorn voice_service.app:main --factory --host 127.0.0.1 --port $VOICE_PORT`, where `main()` builds `VoiceSettings()` and calls `create_app`.

- [ ] **Step 1:** Test `test_health` (`TestClient(create_app(settings_fixture)).get("/voice/health").json() == {"ok": True}`) and `test_cors_allows_web_origin`.
- [ ] **Step 2:** `bash scripts/setup_voice_venv.sh && cd backend/voice && uv run pytest -v` → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Step 2 → PASS.
- [ ] **Step 5:** Commit `feat(voice): voice service skeleton and venv setup`.

---

### Task 4: Qwen3-ASR RealtimeSTT engine and child-process registration (spike first)

**Files:**
- Create: `backend/voice/voice_service/engine_hook.py`, `qwen_engine.py`, `backend/voice/tests/test_qwen_engine.py`, `backend/voice/tests/test_engine_hook.py`

**Interfaces:**
- `ENGINE_NAME = "bangla_qwen3_asr"`. `engine_hook` on import: `factory.ENGINE_CLASS_PATHS.setdefault(ENGINE_NAME, ("voice_service.qwen_engine", "QwenAsrEngine"))`. It must be cheap (no torch import) and never raise (log and continue).
- `QwenAsrEngine(BaseTranscriptionEngine)`: `engine_name = "bangla_qwen3_asr"`; `__init__(config, model_loader=None)` loads `Qwen3ASRForConditionalGeneration.from_pretrained(config.model, dtype=torch.bfloat16, attn_implementation="sdpa")` and `AutoProcessor` (same as qwen-asr `load_model`/`load_processor`) onto `config.device`; `transcribe(audio, language=None, use_prompt=True) -> TranscriptionResult`: language code → `LANGUAGE_NAMES` (default `Bengali`), prompt via `prompt_text(processor, language=name)`, `generation_inputs`, greedy `generate(max_new_tokens=max_new_tokens(min(len(audio)/16000, 30)), do_sample=False)`, decode new tokens, `split_transcript`; `TranscriptionInfo(language=code, language_probability=1.0)`.
- Fallback rule (spec §7.2a): if Step 4's child check fails and no clean hook works, stop and report; do not patch RealtimeSTT.

- [ ] **Step 1 (spike, no GPU):** Test `test_hook_registers_in_fresh_interpreter`: run `[sys.executable, "-c", "from RealtimeSTT.transcription_engines.factory import ENGINE_CLASS_PATHS as E; print('bangla_qwen3_asr' in E)"]` → stdout `True` (proves the `.pth` runs in every process of the venv, including spawned children). Test `test_hook_registers_in_spawned_child`: `multiprocessing.get_context("spawn")` child returns the same check → `True`.
- [ ] **Step 2:** Run → FAIL until the hook exists and `setup_voice_venv.sh` wrote the `.pth`.
- [ ] **Step 3:** Implement `engine_hook.py`; rerun setup script.
- [ ] **Step 4:** Step 1 tests PASS. If the spawned-child test fails, stop and report (spec risk §12).
- [ ] **Step 5:** Engine tests with fakes (no weights): `test_transcribe_uses_language_prefix` (fake processor records the rendered prompt; `language="en"` → endswith `language English<asr_text>`), `test_transcribe_returns_text_after_marker`, `test_unknown_language_defaults_to_bengali`, `test_max_new_tokens_capped_at_30s`. Run → FAIL → implement `qwen_engine.py` → PASS.
- [ ] **Step 6:** Commit `feat(voice): Qwen3-ASR engine registered in RealtimeSTT via .pth hook`.

---

### Task 5: Model listing and CT2 resolution

**Files:**
- Create: `backend/voice/voice_service/models.py`, `backend/voice/tests/test_models.py`

**Interfaces:**
- `@dataclass ModelChoice: id: str; family: Literal["whisper", "qwen3-asr"]; label: str; source: str; origin: Literal["registry", "catalog"]; champion: bool = False; model_version_id: int | None = None`. `id` = `f"registry:{model_version_id}:{export_index}"` or `f"catalog:{entry_id}"`.
- `async list_models(api: httpx.AsyncClient) -> list[ModelChoice]`: registry first (`GET /api/models?task=asr`; per model `GET /api/runs/{run_id}` for `run_dir` and `recipe.backend`): export `format == "ct2"` → whisper, source `{run_dir}/export/{path}`; `format == "safetensors"` and backend `qwen-asr` → qwen3-asr. Then catalog (`GET /api/catalog?task=asr`, `entries`): `runtime.engine == "faster_whisper"` → whisper; `recipe_template.backend == "qwen-asr"` → qwen3-asr; source `hf_repo`. Others skipped. Label registry = `f"{name} v{number}"`.
- `resolve_ct2(source: str, compute_type: str, cache_root: Path, run=subprocess.run) -> Path`: dir with `model.bin` → itself; else cache dir per Global Constraints; convert with `ct2-transformers-converter --model <source> --output_dir <dir> --force --quantization <compute_type> --copy_files preprocessor_config.json` (script next to `sys.executable`, else `python -m ctranslate2.converters.transformers`), then write `tokenizer.json` from `WhisperTokenizerFast.from_pretrained(source).backend_tokenizer.save(...)` if missing. Reference implementation: `whisper-stt-playground/stt_ui.py::resolve_ct2`.

- [ ] **Step 1:** Tests with `httpx.MockTransport`: `test_registry_ct2_is_whisper` (fixture: model 1, run `r-854260e0`, export `ct2` → source `/srv/mlops/runs/r-854260e0/export/ct2`, champion True), `test_registry_qwen_safetensors`, `test_registry_nemo_export_skipped`, `test_catalog_faster_whisper_and_qwen` (`mozilla-ai/whisper-large-v3-turbo-bn`, `SayedShaun/Qwen3-ASR-0.6B-Bengali`; `indicconformer` skipped). `resolve_ct2`: `test_ct2_dir_used_as_is`, `test_cache_path` (`mozilla-ai/whisper-large-v3-turbo-bn`, `float16` → `<root>/mozilla-ai--whisper-large-v3-turbo-bn-float16`), `test_cached_conversion_not_rerun`, `test_converter_command` (fake `run` records argv).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat(voice): list Whisper/Qwen STT models from registry and catalog; CT2 cache`.

---

### Task 6: Voice GPU lock client

**Files:**
- Create: `backend/voice/voice_service/gpu.py`, `backend/voice/tests/test_gpu.py` (sqlite session factory like `backend/conftest.py`)

**Interfaces:**
- `class LockBusy(Exception): holder_kind: str; holder_id: str`; `class RunQueued(Exception): run_id: str`.
- `class VoiceLock(session_factory, holder_id: str, heartbeat_s: float)`: `async recover()` (release any `voice` holder, used at startup); `async acquire()` (raise `RunQueued` if any `Run.state == "queued"`, oldest id; raise `LockBusy` if held by someone else; idempotent for own holder); `async release()`; `async heartbeat_once()`; `start_heartbeat()` / `stop_heartbeat()` (asyncio task calling `heartbeat_once` every `heartbeat_s`).
- `holder_id` = `f"voice-{uuid4().hex[:8]}"` generated per service process.

- [ ] **Step 1:** Tests: `test_acquire_refused_when_run_queued`, `test_acquire_busy_when_run_holds`, `test_acquire_idempotent_for_self`, `test_recover_releases_old_voice_holder`, `test_heartbeat_updates_timestamp`, `test_release_clears`.
- [ ] **Step 2–4:** FAIL → implement using `workers.lock` functions → PASS.
- [ ] **Step 5:** Commit `feat(voice): GPU lock client with heartbeat and queued-run refusal`.

---

### Task 7: Live session: recorder, timing, resampling

**Files:**
- Create: `backend/voice/voice_service/live.py`, `backend/voice/tests/fakes.py` (`FakeRecorder`), `backend/voice/tests/test_live.py`

**Interfaces:**
- `@dataclass Turn: text: str; language: str; model: str; audio_s: float; silence_wait_ms: int; handoff_ms: int; stt_ms: int; speech_end_to_text_ms: int; at: str` (ISO UTC). `Turn.to_event() -> dict` adds `"type": "turn"`.
- `class TurnTimer`: `mark(stage)`, `unmark(stage)`, `has(stage)`, `take() -> dict[str, float]` (returns and clears). Callbacks dict `timing_callbacks(timer, on_state)` exactly as spec §7.3 (`on_turn_detection_start` → `speech_end`; `on_turn_detection_stop` → unmark `speech_end` unless `endpoint`; `on_recording_stop` → `endpoint`; `on_transcription_start` → `stt_start`, returns `None`; `on_recording_start` → state `speaking`).
- `turn_from_marks(marks, done, text, language, model, audio_s) -> Turn | None` (None when text empty or no `speech_end`; spans rounded ms per spec formulas; `endpoint`/`stt_start` fall back to `done` as in the playground).
- `class StreamResampler(sample_rate: int)`: `to_pcm16(frame: bytes) -> bytes` (PCM16 in at `sample_rate` → PCM16 at 16 kHz; identity at 16000; `soxr.ResampleStream(sr, 16000, 1, dtype="float32")`).
- `class LiveSession(recorder_cls, *, model_path, engine, model_label, compute_type, silence_sec, beam_size, language, device, on_event: Callable[[dict], None], turn_log: Path)`: `start()` builds the recorder with the kwargs in spec §7.2 step 2 plus `timing_callbacks` and starts the listen thread (loop `recorder.text()` → `turn_from_marks` → append JSONL → `on_event`; updates `last_activity`); `set_language(code)` sets `recorder.language`; `feed(pcm16_16k: bytes)`; `shutdown()`; property `last_activity: float` (monotonic; set on start, on each turn, and by `touch()`).

- [ ] **Step 1:** Tests with `FakeRecorder` (constructor stores kwargs; `fire(*callbacks)`; `text()` blocks on a queue the test pushes to):
  - `test_spans_match_formulas`: marks 0.0/0.6/0.601/0.9 (speech_end/endpoint/stt_start/done) → `600/1/299/900`.
  - `test_resumed_speech_clears_speech_end`: turn_detection_start then turn_detection_stop before endpoint → no turn.
  - `test_empty_text_dropped_and_idle_unchanged` (Review Focus: silence-only).
  - `test_language_switch_applies_next_turn`: `set_language("en")` → `recorder.language == "en"`; turn records `"en"` (Review Focus).
  - `test_recorder_kwargs`: `post_speech_silence_duration=0.6`, `beam_size=5`, `use_microphone=False`, `enable_realtime_transcription=False`, `language="bn"`, `transcription_engine` passed through.
  - `test_turn_logged_jsonl`.
  - `test_resampler_48k_continuous`: 1 s 440 Hz sine at 48 kHz fed in 0.2 s frames → ~16000 samples out (±200) and max sample-to-sample jump < 0.1 of full scale (no boundary click).
  - `test_resampler_identity_16k`.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat(voice): live session with LocalVoiceAgent turn timing and stream resampling`.

---

### Task 8: HTTP + WebSocket API

**Files:**
- Modify: `backend/voice/voice_service/app.py`
- Create: `backend/voice/voice_service/state.py` (`ServiceState`: current session, loaded model info, settings, lock, idle task), `backend/voice/tests/test_api.py`

**Interfaces (routes):**
- `GET /voice/models` → `[ModelChoice as dict]` (502 with detail when the API is unreachable).
- `POST /voice/load` body `{model_id: str, compute_type: str = "float16", silence_sec: float = 0.6, beam_size: int = 5, language: "bn"|"en" = "bn"}` → `{model, family, convert_s, load_s}`. Order: lock acquire (409 `{"detail": "run r-… is queued"}` / `{"detail": "GPU held by run r-…"}`) → resolve path (whisper: `resolve_ct2` in a thread; qwen: source as is, engine `bangla_qwen3_asr`) → shut down any existing session → `LiveSession.start()` → `lock.start_heartbeat()`. Any failure after acquire releases the lock and returns 500 with the error tail. A concurrent load waits on an `asyncio.Lock` (Review Focus: double load).
- `POST /voice/unload` → `{ok: true}`; stops session, heartbeat, releases lock. Idempotent.
- `GET /voice/status` → `{loaded: ModelChoice|null, family, settings, idle_s, lock: {holder_kind, holder_id}}`.
- `WS /voice/ws`: one client at a time (second gets close code 4409, reason `"another live session is open"`). Text `{"type":"config","language","sample_rate"}` → `set_language`, new `StreamResampler`. Binary → `resampler.to_pcm16` → `session.feed`. Server pushes `state`, `turn`, `error` events (session `on_event` → `asyncio.Queue` → socket via `loop.call_soon_threadsafe`). On disconnect: clear resampler, keep model loaded (Review Focus: mid-turn close). No model loaded → `{"type":"error","detail":"no model loaded"}`.
- Idle task: every 30 s, if loaded and `monotonic() - session.last_activity > idle_unload_s` → unload and log.
- Startup: `lock.recover()`; shutdown: unload.

- [ ] **Step 1:** Tests with `Deps(recorder_cls=FakeRecorder, api_client=mock, lock=in-memory fake)`: `test_load_then_status`, `test_load_409_when_run_queued`, `test_second_load_replaces_first` (first FakeRecorder `shutdown` called, one holder), `test_unload_idempotent`, `test_ws_config_and_turn_event` (send config + 48 kHz frames, push text to FakeRecorder → receive `turn` event with `language`), `test_ws_second_client_rejected`, `test_ws_disconnect_mid_turn_then_reconnect_works`, `test_idle_unload` (idle_unload_s=0.1, fake clock).
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat(voice): load/unload/status API and live WebSocket`.

---

### Task 9: File mode

**Files:**
- Create: `backend/voice/voice_service/filemode.py`, `backend/voice/tests/test_filemode.py`
- Modify: `app.py` (route)

**Interfaces:**
- `class FileTranscriber(family, model_path, device, compute_type)`: lazy-loads on first call: whisper → `faster_whisper.WhisperModel(model_path, device, compute_type)`; qwen → `QwenAsrEngine` from Task 4 (same weights path, in-process). `transcribe(audio: np.ndarray, language: str, vad: bool, beam_size: int) -> str` (whisper: `beam_size`, `vad_filter`, `condition_on_previous_text=False`, segments joined). `close()`.
- `score(hyp: str, ref: str) -> dict` → `{"wer", "cer"}` using `app.core.metrics.wer/cer` on `bn_normalize(..., mode="asr")` texts.
- `POST /voice/transcribe` multipart `file`, `reference=""`, `language="bn"`, `runs=3` (1–10), `vad=true` → `{text, audio_s, latency_ms_mean, latency_ms_best, rtf, wer?, cer?}`; audio via `app.core.backend_kit.audio.load_audio(path, 16000)`; touches session activity; 409 when no model loaded. Unload closes the transcriber.

- [ ] **Step 1:** Tests: `test_score_normalised` (ref `"৳১২৩"` vs hyp `"একশো তেইশ টাকা"` → wer 0.0), `test_transcribe_endpoint_with_fake` (fake transcriber returns fixed text → fields present, rtf = latency/audio_s), `test_runs_bounds` (runs=0 → 422), `test_no_model_409`.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat(voice): file transcription with WER/CER and latency`.

---

### Task 10: Run scripts, config, docs

**Files:**
- Modify: `run-vm.sh`, `.env.example`, `docker-compose.yml`, `README.md`, `scripts/setup_venvs.sh` (call `scripts/setup_voice_venv.sh`)

**Interfaces:**
- `.env.example`: `VOICE_PORT=8010` with comment; `NEXT_PUBLIC_VOICE_URL=http://localhost:8010`.
- `run-vm.sh`: start tmux session `mlops-voice` running the Task 3 command (with `LD_LIBRARY_PATH` = `find backend/voice/.venv/lib -type d -path '*site-packages/nvidia/*/lib'` joined by `:`); `status` shows it and `curl http://localhost:$VOICE_PORT/voice/health`; `stop` stops it before the worker; `dev` adds a `voice` window; port check includes `VOICE_PORT`.
- `docker-compose.yml`: web build arg/env `NEXT_PUBLIC_VOICE_URL`.
- README: "STT test page" section: start, tunnel `ssh -L 3001:localhost:3001 -L 8000:localhost:8000 -L 8010:localhost:8010 <user>@<vm>`, mic needs localhost, GPU lock behaviour.

- [ ] **Step 1:** `bash -n run-vm.sh` → OK. **Step 2:** on the VM `./run-vm.sh status` lists `mlops-voice`, health `{"ok":true}`. **Step 3:** Commit `chore(voice): start the voice service from run-vm.sh; config and docs`.

---

### Task 11: Frontend: voice client, page, settings

**Files:**
- Create: `frontend/src/lib/voice-types.ts`, `frontend/src/lib/voice.ts`, `frontend/src/lib/pcm16.ts`, `frontend/src/lib/turns.ts`, tests `voice.test.ts`, `pcm16.test.ts`, `turns.test.ts`; `frontend/public/pcm16-worklet.js`; `frontend/src/app/test/stt/page.tsx`; `frontend/src/components/stt/model-bar.tsx`, `settings-panel.tsx`, `live-panel.tsx`, `file-panel.tsx`, `turn-table.tsx` (+ `live-panel.test.tsx`)
- Modify: `frontend/src/lib/nav.ts` (group `{ label: "Test", items: [{ href: "/test/stt", label: "STT", icon: Mic }] }`, label map `test: "Test"`, `stt: "STT"`), `frontend/src/lib/palette.ts` (entry), `nav.test.ts`

**Interfaces:**
- `voice-types.ts`: `ModelChoice`, `VoiceStatus`, `TurnEvent` (fields = Task 7 `Turn` + `type`), `StateEvent`, `ErrorEvent`, `Language = "bn" | "en"`.
- `voice.ts`: `VOICE_URL = process.env.NEXT_PUBLIC_VOICE_URL ?? "http://localhost:8010"`; `listVoiceModels()`, `loadModel(body)`, `unloadModel()`, `getVoiceStatus()`, `transcribeFile(form)`, `openLiveSocket(onEvent): LiveSocket` with `sendConfig(language, sampleRate)`, `sendPcm(buf: ArrayBuffer)`, `close()`. Errors reuse `ApiError`.
- `pcm16.ts`: `floatToPcm16(f: Float32Array): Int16Array` (clamp ±1, ×32767). Worklet posts ~50 ms Int16 buffers; capture via `new AudioContext({ sampleRate: 16000 })`, falling back to the default context and sending its real `sampleRate` in config.
- `turns.ts`: `turnsReducer(state, action)` actions `turn | clear`; keeps newest first, max 200; `toCsv(turns): string` header `at,model,language,audio_s,silence_wait_ms,handoff_ms,stt_ms,speech_end_to_text_ms,text`.
- Settings: language persisted under `localStorage["stt.language"]` (try/catch), default `bn`; change → `sendConfig` immediately when live; silence/beam show "reload to apply".
- Page states: service down (banner: "Voice service not running — start it with `./run-vm.sh` (tmux session `mlops-voice`)", controls disabled, poll `getVoiceStatus` every 3 s, recover automatically); lock held by run (Load disabled, holder shown); run queued while voice loaded ("Run r-… is queued: unload to start it", using `getSystem()` `queue.queued` and `lock.holder_kind === "voice"`); mic denied ("Microphone blocked: open the page through the SSH tunnel on localhost").
- Last-turn line copy: `speech end → text {total} ms = silence {s} + hand-off {h} + STT {t}`.
- `?model=<id>` query preselects the model.

- [ ] **Step 1:** Tests: `floatToPcm16` clamps and scales (`[1.5,-1.5,0.5] → [32767,-32767,16384]` ±1); reducer newest-first and cap 200; CSV header and quoting of commas in Bangla text; nav has `/test/stt`; `live-panel.test.tsx`: renders last-turn copy from a `TurnEvent`; shows service-down banner when status fetch rejects and clears it when it resolves (Review Focus).
- [ ] **Step 2:** `cd frontend && npm test` → FAIL. **Step 3:** Implement. **Step 4:** `npm test && npm run typecheck && npm run build` → PASS.
- [ ] **Step 5:** Commit `feat(web): STT live test page with language switch and file mode`.

---

### Task 12: Registry "Test" link and Runs-page unload notice

**Files:**
- Modify: `frontend/src/app/registry/[id]/page.tsx`, `frontend/src/app/runs/page.tsx`, `frontend/src/app/runs/[id]/page.tsx`
- Create: `frontend/src/components/voice-lock-notice.tsx` (+ test)
- Create: `frontend/src/lib/stt-support.ts` (+ test)

**Interfaces:**
- `isSttTestable(model: ModelVersion, runBackend?: string): boolean` → task `asr` and (an export `format === "ct2"` or (`format === "safetensors"` and backend `qwen-asr`)). Registry page shows a "Test" link to `/test/stt?model=registry:<id>:<exportIndex>` when true.
- `<VoiceLockNotice system={SystemInfo} />`: when `lock.holder_kind === "voice"` and `queue.queued > 0` → "GPU held by STT test" + **Unload** button calling `unloadModel()` from `voice.ts`; on failure shows "Voice service unreachable". Rendered on the Runs list and on a queued run's detail page.

- [ ] **Step 1:** Tests: `isSttTestable` cases (ct2 true; safetensors+qwen-asr true; safetensors+hf false; nemo false; task tts false); notice renders only for voice holder with queued runs; Unload calls `unloadModel`.
- [ ] **Step 2–4:** `npm test` FAIL → implement → PASS; typecheck, build.
- [ ] **Step 5:** Commit `feat(web): registry Test link; unload notice for queued runs`.

---

### Task 13: Heavy end-to-end check on the VM

**Files:**
- Create: `backend/voice/tests/test_heavy_e2e.py` (`@pytest.mark.heavy`)

**Interfaces:** uses the real app (`create_app` with real deps, sqlite DB, `api_client` mock returning catalog entries), FLEURS bn_in test clip from `~/.cache/huggingface/hub/datasets--google--fleurs/snapshots/*/data/bn_in/audio/test.tar.gz` (skip if absent), upsampled to 48 kHz, fed over `/voice/ws` in 0.2 s frames at real-time pace plus 2 s silence.

- [ ] **Step 1:** `test_whisper_bn_live`: load `catalog:whisper-large-v3-turbo-bn`; one `turn` event; `cer(normalised ref, hyp) < 0.2`; all four spans present, `silence_wait_ms >= 550`.
- [ ] **Step 2:** `test_qwen_bn_live`: same with `catalog:qwen3-asr-0-6b-bn` (engine `bangla_qwen3_asr` in the recorder child).
- [ ] **Step 3:** `test_language_switch_en`: after config `en`, the next turn event has `language == "en"`.
- [ ] **Step 4:** Run on the VM: `cd backend/voice && uv run pytest -m heavy -v` → PASS (record timings in the commit message).
- [ ] **Step 5:** Commit `test(voice): heavy live e2e for Whisper and Qwen3-ASR`.
