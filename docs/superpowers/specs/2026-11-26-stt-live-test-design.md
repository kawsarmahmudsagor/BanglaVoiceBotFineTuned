# STT live test page: design

Date: 2026-11-26
Status: draft, for later. Approach A chosen by the owner (2026-11-26).

## 1. Intent

The owner fine-tunes Whisper large-v3-turbo for Bangla (first result: model version
`training-1-whisper-v3-turbo-bn`, run `r-854260e0`, base `mozilla-ai/whisper-large-v3-turbo-bn`, WER 48.96% → 32.63%,
`ct2` export). Today the only way to try a checkpoint by voice is the standalone Gradio tool in
`stt-playground/`. The platform should have a built-in page to test an STT model the way LocalVoiceAgent
runs it: speak, the turn ends after a short silence, the transcript appears automatically, and every turn shows
where the latency went.

This is the STT slice of the main spec's Playground (§7) and Voice call (§8) in
`2026-11-23-bangla-voice-mlops-design.md`. LLM and TTS come later, after TTS fine-tuning, and will build on the
service introduced here.

## 2. Goals

1. Pick any supported STT model, from two families:
   - **Whisper**: a registry model version with a `ct2` export, or a catalog base model whose runtime engine is
     `faster_whisper` (`openai/whisper-large-v3-turbo`, `mozilla-ai/whisper-large-v3-turbo-bn`);
   - **Qwen3-ASR**: a registry model version from the `qwen-asr` backend (`safetensors` export, served with
     transformers), or a catalog entry whose recipe backend is `qwen-asr` (`SayedShaun/Qwen3-ASR-0.6B-Bengali`).
2. **Live mode**: browser mic → RealtimeSTT recorder → automatic transcription at turn end, with per-turn
   latency: silence wait, hand-off, STT, and speech end → text.
3. **Language selector** (English, Bangla) in the page's settings, switchable while live. It takes effect from the
   next turn, with no model reload.
4. **File mode**: upload or record a clip, optionally paste a reference, get transcript, latency, RTF, WER and CER
   (both texts through `bn_normalize(mode="asr")`, as the evaluation does).
5. Respect the GPU lock: a loaded test model holds it; training waits; nothing is preempted.

## 3. Non-goals

- LLM, TTS, barge-in, diarization, RAG (the later call page).
- The language selector changes STT only. LLM reply language and TTS voice hook into the same setting later.
- Comparing up to 3 models side by side (spec §7). One model is loaded at a time; the turn table records which
  model produced each row, so loading another model and repeating gives the comparison.
- NeMo (IndicConformer, BanglaConformer) and other STT engines. The service selects the engine per model family so
  they can be added later.
- Storing turns in Postgres or a latency leaderboard. Turns go to a JSONL log only.
- Auth (the platform has none; opened through the SSH tunnel).

## 4. Anti-goals (would make it a failure even if it works)

- Latency numbers that are not comparable with LocalVoiceAgent's. Use the same recorder, the same turn-end rule
  (`post_speech_silence_duration`, default 0.6 s = LocalVoiceAgent `STT_SILENCE_SEC`) and the same stage marks
  (its `stt.timing_callbacks`).
- Garbled audio from resampling. RealtimeSTT's `feed_audio` resamples each chunk independently with scipy's FFT
  `resample`; with 48 kHz browser audio in 0.2 s chunks this produced gibberish (reproduced on 2026-11-26 with a
  FLEURS bn_in clip: gibberish at 48 kHz, correct at 16 kHz). Audio must reach the recorder as continuous 16 kHz
  mono PCM16.
- A test model silently blocking training, or a crashed service leaving the GPU lock held forever.

## 5. Constraints

- Runs on the Ubuntu VM (RTX 5090, `sm_120`, `torch==2.7.1+cu128`). The browser is on the owner's PC, through the
  SSH tunnel; the mic works only on `localhost` or HTTPS.
- The API runs in Docker without GPU compute (`NVIDIA_DRIVER_CAPABILITIES: utility`); GPU work runs natively, like
  the worker.
- Do not edit RealtimeSTT source (LocalVoiceAgent `VENDORED.md` rule).
- Tests run without GPU, network, Docker or model weights (project rule).
- Bangla UI text is not required; labels in English.

## 6. Approaches

**A. Native voice service (recommended).** New uv project `backend/voice/` (FastAPI + RealtimeSTT + faster-whisper),
started natively by `run-vm.sh` in tmux on its own port. The browser talks to it directly (REST + WebSocket), as it
already does with the API. Trade-off: one more process and one more tunnelled port.

**B. Worker-managed serve job (closest to spec §7 wording).** The API enqueues a serve job; the native worker
starts `mlops_backend serve` in the hf backend venv; the API proxies the WebSocket. Trade-off: about twice the work
(new long-lived job type in a worker built for one run at a time, host route from the API container, WebSocket
proxy), and every audio chunk takes an extra hop.

**C. Embed the Gradio playground in an iframe.** Fastest, but outside the platform: no registry picker, no GPU
lock, Gradio streaming adds latency. Rejected: works against having it built in.

Chosen by the owner: **A**.

## 7. Design (approach A)

### 7.1 Voice service

- Location `backend/voice/`, package `voice_service`, own `pyproject.toml` and venv (same pattern as
  `backend/workers/backends/*`: depends on `bangla-voice-backend` by path for `app.core.text.bn_normalize` and the
  lock helpers; excluded from the uv workspace).
- Dependencies: `fastapi`, `uvicorn[standard]`, `RealtimeSTT[faster-whisper]`, `faster-whisper`, `ctranslate2`,
  `transformers==5.24.1` (the qwen-asr backend's pin: Qwen3-ASR inference needs transformers 5; the Whisper HF → CT2
  conversion was verified with 5.24.1 in `stt-playground/`), `accelerate`, `librosa`, `jinja2` (Qwen chat
  template), `soxr`, `torch==2.7.1` (cu128 index), DB driver used by the worker. Not the `qwen_asr` pip package.
- RealtimeSTT comes from LocalVoiceAgent as a pinned git dependency
  (`git+https://github.com/kawsarmahmudsagor/LocalVoiceAgent@fd6bb52258a3154a20d75249ee803dcd99dc3df7#subdirectory=RealtimeSTT`),
  which is what `stt-playground/` uses and was verified with. (Spec §8.1 plans a vendored copy for the call
  page; switching to it then is a one-line source change.)
- Port `VOICE_PORT` (default 8010), host `127.0.0.1`. `run-vm.sh` starts it in tmux session `mlops-voice` and
  `status`/`stop` cover it; `run-vm.sh dev` starts it too. `.env.example` gains `VOICE_PORT`; the frontend gains
  `NEXT_PUBLIC_VOICE_URL` (default `http://localhost:8010`, same pattern as `NEXT_PUBLIC_API_URL`). CORS allows
  `WEB_ORIGIN`. README tunnel command adds `-L 8010:localhost:8010`.
- CUDA libraries: set `LD_LIBRARY_PATH` to the venv's `site-packages/nvidia/*/lib` dirs before start
  (CTranslate2 needs cuBLAS/cuDNN 12; `stt-playground/run.sh` does the same).
- The recorder spawns a child process; the service module must not build heavy state at import.

### 7.2 Model sources and loading

- `GET /voice/models` returns each model with its family (`whisper` | `qwen3-asr`) and source:
  - registry: `GET {API}/api/models?task=asr`; path = `{run_dir}/export/{export.path}` (run dir and backend from
    `GET {API}/api/runs/{run_id}`). Label = model version name + number, champion marked.
    - export `format == "ct2"` → `whisper`;
    - export `format == "safetensors"` from a run whose backend is `qwen-asr` → `qwen3-asr`.
  - catalog: `GET {API}/api/catalog`, `task == asr`:
    - `runtime.engine == faster_whisper` → `whisper`, source `hf_repo`;
    - `recipe_template.backend == qwen-asr` → `qwen3-asr`, source `hf_repo` (catalog `runtime.engine` is `custom`).
  - Anything else (NeMo exports, other engines) is not listed.
- Whisper: loading a catalog (HF-format) model converts it once to CT2 with `ct2-transformers-converter --quantization
  <compute_type> --copy_files preprocessor_config.json`, then writes `tokenizer.json` from `WhisperTokenizerFast` if
  the repo lacks it (`mozilla-ai/whisper-large-v3-turbo-bn` lacks it). Cache: `$DATA_ROOT/models/ct2-cache/<repo with
  / as -->-<compute_type>/`. A dir with `model.bin` is used as is.
- `POST /voice/load {source, compute_type, silence_sec, beam_size, language}`:
  1. acquire the GPU lock (`holder_kind="voice"`, `holder_id=<service instance id>`); if held by a run → 409 with
     the holder, the page shows "GPU busy: run r-…";
  2. build `AudioToTextRecorder(model=<ct2 dir | qwen model dir or repo>, transcription_engine=<"faster_whisper" |
     the Qwen engine name, §7.2a>, device="cuda",
     compute_type, language, use_microphone=False, spinner=False, enable_realtime_transcription=False,
     post_speech_silence_duration=silence_sec, beam_size, **timing callbacks)`;
  3. warm up. Response: load time, conversion time.
  On failure, release the lock.
- `POST /voice/unload` shuts the recorder down and releases the lock. Idle unload after 15 min without a finished
  speech turn and without a file request (spec §3.3). Incoming WebSocket audio frames do not reset the idle clock:
  the browser streams frames during silence too, so an open page with the mic on must not hold the GPU forever.
- `GET /voice/status`: loaded model, settings, lock holder, idle seconds.
- Lock hygiene:
  - on startup the service releases any lock held with `holder_kind="voice"` (a previous instance crashed); on
    shutdown it releases its own;
  - **heartbeat**: new nullable column `gpu_lock.heartbeat_at` (Alembic migration). The voice service updates it
    every 10 s while it holds the lock. The worker's GPU tick releases a `voice` lock whose `heartbeat_at` is older
    than 60 s (service killed by OOM/SIGKILL/segfault, which skips the shutdown hook; `run-vm.sh` keeps dead tmux
    panes and does not restart it). Run locks are unaffected.
- Queued training (owner's policy, "wait and make it visible"):
  - training is never preempted: queued runs wait while the lock is held (`claim_next_run` returns nothing);
  - `POST /voice/load` is refused (409, "run r-… is queued") while any run is `queued`, so a test model cannot grab
    the lock between worker polls ahead of a queued run;
  - the test page shows "Run r-… is queued: unload to start it" while a run waits on a voice-held lock;
  - the Runs page (and the run detail of a queued run) shows "GPU held by STT test" with an **Unload** button.
    The API needs a way to ask the voice service to unload: the button calls `POST {VOICE_URL}/voice/unload` from
    the browser (the API container cannot reach the native service without a host route); the API's system
    endpoint exposes the lock holder (and `heartbeat_at`) so the page knows when to show it.
- Changing `silence_sec` or `beam_size` needs a reload (recorder init args). `language` does not (§7.4).

### 7.2a Qwen3-ASR engine

- Inference must match how the `qwen-asr` backend trains and evaluates, or fine-tunes will look worse than they are:
  `Qwen3ASRForConditionalGeneration` + `AutoProcessor` (transformers 5), bf16, greedy (`do_sample=False`), the
  model card's chat format with the assistant turn prefilled with `language <Name><asr_text>`, `max_new_tokens`
  from the clip length, transcript = text after the marker. Reuse the backend's code
  (`backend/workers/backends/qwen-asr/mlops_backend/prompt.py`, `qwen_asr.load_model`, `load_processor`,
  `make_transcriber`) by import or by moving the shared, torch-free parts into `app.core.backend_kit`; do not
  fork the prompt.
- RealtimeSTT's own `qwen3_asr` engine is **not** used: it wraps the `qwen_asr` pip package (`Qwen3ASRModel`), a
  different inference path from the backend, and its language table has no Bengali.
- The service adds its own engine class subclassing RealtimeSTT's `BaseTranscriptionEngine`
  (`transcription_engines/base.py`; `transcribe(audio, language=None, use_prompt=True) -> TranscriptionResult`),
  as main spec §8.2 planned for non-Whisper engines.
- Registration constraint (verified by reading RealtimeSTT, not run): engines are looked up in a fixed dict,
  `ENGINE_CLASS_PATHS` in `transcription_engines/factory.py`, and the engine is built inside the recorder's
  **spawned child process**. Mutating the dict in the parent does not reach the child. The registration must run
  at import time in the child, without editing RealtimeSTT (for example a module the child is guaranteed to import,
  or a `.pth` hook in the voice venv). Spike this first; if no clean hook exists, fall back to running the Qwen
  model in the service process with the same turn-end rule, and record the deviation.
- Language: `bn` → `Bengali`, `en` → `English` in the prefix. The fine-tunes are Bengali-only; English output is
  whatever the model does and is reported as such.
- VRAM: about 1.5 GB in bf16 for 0.6B; no conversion step.

### 7.3 Live mode protocol

WebSocket `/voice/ws`. One live session at a time (single operator); a second connection gets closed with a reason.

Client → server:
- text `{"type": "config", "language": "bn" | "en", "sample_rate": 16000}`: sent on connect and whenever the
  language changes;
- binary frames: mono PCM16 little-endian at `sample_rate`, any size (the client sends ~20–100 ms).

Server → client:
- `{"type": "state", "value": "listening" | "speaking" | "transcribing"}` from recorder callbacks;
- `{"type": "turn", "text", "language", "model", "audio_s", "silence_wait_ms", "handoff_ms", "stt_ms",
  "speech_end_to_text_ms", "at"}`;
- `{"type": "error", "detail"}`.

Server audio path: if `sample_rate != 16000`, resample with a per-connection `soxr.ResampleStream` (stateful), then
`recorder.feed_audio(pcm16_bytes, original_sample_rate=16000)`. Never let RealtimeSTT resample.

A listen thread loops `recorder.text()` and emits turns. Stage marks, as in LocalVoiceAgent `timing_callbacks`:
- `speech_end` ← `on_turn_detection_start`; cleared by `on_turn_detection_stop` unless `endpoint` is set;
- `endpoint` ← `on_recording_stop`;
- `stt_start` ← `on_transcription_start` (must return `None`; truthy aborts);
- `stt_done` ← `text()` returned.
Spans: `silence_wait_ms = endpoint − speech_end`, `handoff_ms = stt_start − endpoint`,
`stt_ms = stt_done − stt_start`, `speech_end_to_text_ms = stt_done − speech_end`. Empty transcripts or turns without
`speech_end` are dropped. Each turn is appended to `$DATA_ROOT/logs/voice-turns.jsonl` with model and settings.

### 7.4 Language selector

- Values: English → `en`, Bangla → `bn`. Default Bangla.
- Live: the `config` message sets `recorder.language`; RealtimeSTT sends `recorder.language` with every final
  transcription request (`core/transcription_api.py`), so it applies from the next turn without reloading. Verified by
  reading the code, not yet run.
- File mode passes it as `language=` to the transcription call.
- Qwen3-ASR maps the code to the prefix language name (§7.2a).
- Every turn row shows the language it was transcribed with.
- Stored in the browser (`localStorage`) so it survives reloads; it lives in the page's Settings panel, designed
  so LLM/TTS can read the same value later.

### 7.5 File mode

`POST /voice/transcribe` (multipart: `file`, optional `reference`, `language`, `runs` 1–10, `vad` bool): decode
with `app.core.backend_kit.audio.load_audio` (ffmpeg fallback for webm/m4a), transcribe through the loaded model
(the recorder's model is in its child process; the service may hold a second `faster_whisper.WhisperModel` on the
same CT2 dir, loaded lazily on first file request, ~1.6 GB VRAM extra; builder's call if a cleaner route exists).
Returns text, audio seconds, mean/best latency, RTF, and WER/CER when a reference is given.

### 7.6 Page

Route `/test/stt` ("Test → STT" in the sidebar group; command palette entry). The model picker groups models by
family (Whisper, Qwen3-ASR); compute type applies to Whisper only (Qwen runs bf16). Layout:

```
[Model ▾ (registry ct2 / catalog)] [Compute ▾ float16] [Load] [Unload]   GPU: voice (idle 2m) | busy: run r-…
Settings: Language (Bangla | English)   Turn-end silence 0.6 s   Beam 5   (silence/beam: "reload to apply")
[Live] [File]
 Live:  [● Start mic] [■ Stop]   state: listening / speaking / transcribing
        Last transcript (large)
        Last turn: speech end → text 1084 ms = silence 782 + hand-off 4 + STT 299
        Turns table: time | model | lang | speech s | silence ms | hand-off ms | STT ms | total ms | text
 File:  [upload / record] [reference text] [runs 3] [VAD ✓] [Transcribe] → text, latency, RTF, WER, CER
```

Mic capture: `getUserMedia` → `AudioContext({ sampleRate: 16000 })` → AudioWorklet that converts Float32 to PCM16
and posts ~50 ms frames → WebSocket. If the browser refuses a 16 kHz context, capture at the native rate and send
that `sample_rate` in `config` (server resamples with the stateful stream). The turn table is in-memory per page
plus a "download CSV" button.

Model detail page (`/registry/[id]`) gets a "Test" link to `/test/stt?model=<id>` for ASR models the service
supports (`ct2` export, or `safetensors` from the `qwen-asr` backend).

### 7.7 Errors

- Voice service down → page banner with the start command (`./run-vm.sh` or tmux session name).
- Lock held by a run → load disabled with the holder shown; the page polls status.
- Model load failure (conversion, CUDA libs) → message with the stderr tail.
- Mic permission denied / not on localhost → explain the tunnel requirement.

## 8. Testing

- No GPU: the recorder class is injectable (as LocalVoiceAgent `make_recorder(recorder_cls=…)`); a fake recorder
  drives callbacks and `text()` to test span maths, dropped turns, language switching, and the WebSocket protocol
  (FastAPI `TestClient`).
- Resampling: feeding a 48 kHz sine in 0.2 s chunks through the stream resampler gives a continuous 16 kHz signal
  (no boundary discontinuity above a threshold).
- Model listing against a mocked API (registry + catalog): family detection for ct2, qwen-asr safetensors, catalog
  `faster_whisper` and `qwen-asr` entries; NeMo exports excluded. CT2 cache path, tokenizer.json fallback.
- Qwen engine: prompt and transcript extraction with a fake processor/model (no weights), matching the backend's
  `prompt.py` behaviour; language name mapping.
- Lock: acquire/release/startup recovery, heartbeat expiry reclaimed by the worker tick, load refused while a run
  is queued, against the test database the backend tests already use.
- Frontend: vitest for the page reducer (turn rows, state), settings persistence, worklet PCM16 conversion.
- Heavy (VM, `--heavy`): load `mozilla-ai/whisper-large-v3-turbo-bn`, feed a FLEURS bn_in clip upsampled to
  48 kHz through the WebSocket, expect a transcript close to the reference (CER < 0.2) and all spans present. Same
  for `SayedShaun/Qwen3-ASR-0.6B-Bengali` through the custom engine.

## 9. Decided by the owner

- Scope now: STT live test page only; call page (LLM/TTS) later.
- Language selector affects STT only, switchable in real time.
- Approach A (native voice service).
- Qwen3-ASR is in scope alongside Whisper (catalog `qwen3-asr-0-6b-bn` and `qwen-asr` backend exports).
- Queued training while a test model is loaded: run waits, made visible on the test and Runs pages with an
  Unload button; Load refused while a run is queued.
- Turn-end timing follows LocalVoiceAgent (`STT_SILENCE_SEC` 0.6 s, RealtimeSTT recorder).

## 10. Author's calls (open to correction)

- Port 8010, route `/test/stt`, default language Bangla.
- RealtimeSTT as a pinned git dependency rather than a vendored copy, for now.
- One model at a time; comparison by reloading.
- Turns logged to JSONL, not the database.
- Qwen inference reuses the qwen-asr backend's code path, not RealtimeSTT's `qwen3_asr` engine; registered as a
  custom engine in the recorder's child, with an in-process fallback if no clean hook exists.
- Voice venv on transformers 5.24.1 for both families.
- Stale voice-lock recovery via a heartbeat column reclaimed by the worker (touches schema and worker).
- Idle = no finished speech turn and no file request for 15 min; audio frames do not count.

## 11. Left to the builder

- Exact component structure, styling (follow existing shadcn patterns), CSV format.
- Whether file mode reuses the recorder's model or a second in-process model.
- Idle-timer implementation.
- Whether a long load/conversion runs synchronously or as a background task with status polling.
- WebSocket reconnect handling; the `NEXT_PUBLIC_VOICE_URL` build arg in compose.

## 12. Risks

- Registering a custom engine inside RealtimeSTT's spawned child (§7.2a) is unverified; it is the first task to
  spike.
- transformers 5.24.1 plus `ct2-transformers-converter` worked for Whisper large-v3-turbo in the playground; other
  Whisper variants are untested.

## 13. Relation to `stt-playground/`

The standalone Gradio tool stays as a quick offline check until this page ships; its verified pieces (CT2
conversion with tokenizer fallback, stateful resampling, timing callbacks) are the reference implementation for
§7.2–7.3.
