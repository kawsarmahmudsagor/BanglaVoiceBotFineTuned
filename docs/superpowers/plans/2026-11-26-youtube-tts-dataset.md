# YouTube TTS Dataset Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone tool that turns YouTube videos of one Bangla speaker into a stage-B TTS dataset (`metadata.csv` + `wavs/` + `curation.json`), with a web UI that shows live progress and a review queue.

**Architecture:** A FastAPI backend owns a SQLite store, a job runner (a CPU/network thread pool plus one GPU worker that runs stages model by model), and a WebSocket event bus. Pipeline stages are separate modules with pure decision functions (segmenting, routing, splits, punctuation guard) kept apart from model adapters, so the decisions are unit-tested without models. A Vite + React + TypeScript frontend talks to REST and one WebSocket per job.

**Tech Stack:** Python 3.12, uv, FastAPI, uvicorn, sqlite3 (stdlib), torch 2.7.1 cu128, transformers 5.24.1, faster-whisper, demucs, silero-vad, yt-dlp, torchmetrics[audio] (DNSMOS), speechbrain (ECAPA), httpx; Vite, React 18, TypeScript, Vitest, Testing Library.

**Spec:** `docs/superpowers/specs/2026-11-26-youtube-tts-dataset-design.md`

## Global Constraints

- Target machine: Windows 11, RTX 4060 8 GB, Windows PowerShell 5.1. Every dependency must install on Windows; `[tool.uv] environments = ["sys_platform == 'win32'"]`. No flash-attn; attention is SDPA.
- Scripts are PowerShell 5.1: no `&&`, `||`, ternary or `??`.
- One model on the GPU at a time. Peak VRAM under 7 GB.
- Sub-project root: `youtube-tts-dataset/`. Work data under `youtube-tts-dataset/var/` (already git-ignored by the root `var/` rule).
- Config lookup order: environment, `youtube-tts-dataset/.env`, repo root `.env`; first value wins.
- Defaults: `OLLAMA_HOST=http://localhost:11434`, `LLM_MODEL=gemma4:31b-cloud` (slug verified in Task 8), `YTDS_PORT=8765`, `YTDS_EXPORT_DIR=<repo>/var/import`.
- Threshold defaults: `cer_auto=0.05`, `cer_fallback=0.30`, `dnsmos_min=3.0` (OVRL), `speaker_sim_min=0.60`, `clip_min_s=2.0`, `clip_max_s=15.0`, `merge_gap_s=0.30`, `demucs_segment_s=7.0`.
- Export audio: trim with 150 ms padding, −23 LUFS, −1 dBTP, 24 kHz, 16-bit PCM WAV, mono. `metadata.csv` columns exactly `file,text,speaker,split`; `file` is `wavs/<name>.wav`.
- Splits 90/5/5 by duration; videos with under 60 s kept go to train only.
- Files on disk are named by YouTube video id and clip index only, never by title.
- Text comparisons use `app.core.text.bn_normalize(text, mode="asr")`.
- Models: A = `SayedShaun/Qwen3-ASR-0.6B-Bengali`; B = `mozilla-ai/whisper-large-v3-turbo-bn` (CT2 int8_float16); fallback = `large-v3` faster-whisper `task="translate"`; separation = demucs `htdemucs`; VAD = silero-vad; DNSMOS via `dataprep.steps.dnsmos.score_dnsmos`; ECAPA via `dataprep.steps.speakers.make_embedder`.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. The same video pasted twice, or present both as a URL and inside a pasted playlist: deduplicate by video id within a dataset, with one row. Test in Task 4.
2. A video that is all music or has no speech: zero clips, the video ends `done` with a "<30 s clean audio" warning, export ignores it. Test in Task 9.
3. STT output that is empty, or holds Latin words or digits: routing must not crash on an empty reference, and an empty model A routes to `fallback`. Test in Task 7.
4. Backend restarted mid-job: unfinished stages rerun, done stages are skipped, the WebSocket client reconnects and refetches state. Tests in Task 9 (runner) and Task 12 (hook).
5. Audio seeking in the review card: the clip endpoint must answer HTTP Range requests (206), or the browser cannot seek. Test in Task 11.

---

## File Structure

```
youtube-tts-dataset/
  README.md  .env.example  run.ps1
  backend/
    pyproject.toml
    ytds/
      __init__.py
      config.py          Settings, .env lookup, Thresholds
      store.py           SQLite schema and access
      events.py          EventBus with per-video throttle
      gpu.py             GpuWorker: one model at a time, OOM retry
      llm.py             Ollama client: show, punctuate, translate
      runner.py          JobRunner: stage graph, resume, dispatch
      splits.py          assign_splits
      exporter.py        export_dataset
      pipeline/
        __init__.py      Stage names and order
        download.py      yt-dlp wrapper
        separate.py      demucs vocals
        segment.py       VAD regions -> clips
        quality.py       DNSMOS + speaker filter
        transcribe.py    model A / B adapters
        routing.py       cer, route
        punct.py         punctuation guard
        fallback.py      translate path
      api/
        __init__.py      create_app
        jobs.py  review.py  dataset.py  settings.py  ws.py
    tests/ (one test file per module, conftest.py with fakes)
  frontend/
    package.json  vite.config.ts  tsconfig.json  index.html
    src/
      main.tsx  App.tsx  api.ts  types.ts  useJobEvents.ts  theme.css
      pages/Jobs.tsx  pages/Review.tsx  pages/Dataset.tsx  pages/Settings.tsx
      components/ProgressRow.tsx  components/ReviewCard.tsx  components/Waveform.tsx
      __tests__/ProgressRow.test.tsx  __tests__/ReviewCard.test.tsx  __tests__/useJobEvents.test.ts
```

Shared types (Python, `ytds/store.py`; mirrored in `frontend/src/types.ts`):

- `Stage = Literal["download","separate","segment","quality","transcribe","fallback","punctuate"]` in that order (`pipeline/__init__.py: STAGES`).
- `VideoStatus = Literal["queued","running","done","failed","skipped"]`.
- `ClipStatus = Literal["dropped","auto","flagged","needs_review","pending_llm","approved","rejected"]`.
- `TextSource = Literal["stt","translated"]`.
- `DropReason = Literal["length","dnsmos","other_speaker"]`.

---

### Task 1: Backend scaffold and config

**Files:**
- Create: `youtube-tts-dataset/backend/pyproject.toml`, `ytds/__init__.py`, `ytds/config.py`, `youtube-tts-dataset/.env.example`
- Test: `youtube-tts-dataset/backend/tests/test_config.py`

**Interfaces:**
- Produces: `load_settings(env: Mapping[str,str] | None = None, sub_env: Path | None = None, root_env: Path | None = None) -> Settings`; `Settings` (pydantic) fields `ollama_host, ollama_api_key, llm_model, data_dir: Path, export_dir: Path, port: int, hf_token`; `Thresholds` (pydantic) with the Global Constraints defaults; `SUB_ROOT`, `REPO_ROOT: Path`.

- [ ] **Step 1: pyproject.** `name = "ytds"`, `requires-python = ">=3.12,<3.13"`. Dependencies: `bangla-voice-backend[audio]`, `bangla-dataprep`, `torch==2.7.1`, `torchaudio==2.7.1`, `transformers==5.24.1`, `accelerate`, `faster-whisper`, `ctranslate2`, `demucs`, `silero-vad`, `yt-dlp`, `torchmetrics[audio]`, `speechbrain>=1.0`, `pyloudnorm`, `soxr`, `soundfile`, `librosa`, `fastapi>=0.110`, `uvicorn[standard]`, `httpx`, `python-dotenv`, `pydantic>=2`. Dev group: `pytest`, `pytest-asyncio`. `[tool.uv.sources]`: `bangla-voice-backend = { path = "../../backend", editable = true }`, `bangla-dataprep = { path = "../../tools/dataprep", editable = true }`, torch/torchaudio from index `pytorch-cu128` (`https://download.pytorch.org/whl/cu128`, explicit). `[tool.uv] environments = ["sys_platform == 'win32'"]`. pytest markers `gpu`, `network`; `addopts = "-m 'not gpu and not network'"`.
- [ ] **Step 2: Run `uv sync` in `youtube-tts-dataset/backend`.** Expected: resolves and installs on Windows. If `bangla-dataprep` base deps fail on Windows, stop and report the failing package.
- [ ] **Step 3: Write failing tests** `test_env_wins_over_files`, `test_sub_env_wins_over_root_env`, `test_root_env_used_when_sub_missing`, `test_defaults` (asserts `llm_model == "gemma4:31b-cloud"`, `port == 8765`, `Thresholds().cer_auto == 0.05`, `Thresholds().cer_fallback == 0.30`), using `tmp_path` `.env` files and an explicit `env` mapping.
- [ ] **Step 4: Run `uv run pytest tests/test_config.py -v`.** Expected: FAIL (import error).
- [ ] **Step 5: Implement `config.py`.** Read both files with `dotenv_values`; merge as root < sub < env.
- [ ] **Step 6: Run the tests.** Expected: PASS.
- [ ] **Step 7: `.env.example`** lists every variable in spec §2.1 with its default and a comment that `OLLAMA_API_KEY` is only needed when `OLLAMA_HOST=https://ollama.com`.
- [ ] **Step 8: Commit** `feat(ytds): backend scaffold and settings lookup`.

### Task 2: SQLite store

**Files:**
- Create: `ytds/store.py`, `ytds/pipeline/__init__.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Produces: `Store(path: Path)` (thread-safe; one `sqlite3` connection with `check_same_thread=False` and a `threading.Lock`; WAL mode). Tables:
  - `datasets(id TEXT PK, name TEXT UNIQUE, speaker TEXT, created REAL)`
  - `videos(dataset_id, id /*yt id*/, url, title, channel, upload_date, duration_s REAL, license TEXT, status, error TEXT, warning TEXT, PRIMARY KEY(dataset_id, id))`
  - `stages(dataset_id, video_id, stage, done INTEGER, updated REAL, PRIMARY KEY(dataset_id, video_id, stage))`
  - `clips(dataset_id, id /*"<video>_<0000>"*/, video_id, idx INT, PRIMARY KEY(dataset_id, id), start_s, end_s, dnsmos REAL, speaker_sim REAL, drop_reason TEXT, text_a, text_b, cer REAL, text_en, text_translated, text_final, text_source, status, no_punct INT DEFAULT 0)`
  - `settings(key TEXT PK, value TEXT)` holding `Thresholds` JSON and per-dataset speaker.
- Methods: `create_dataset(name, speaker) -> str`, `add_video(dataset_id, video_id, url) -> bool` (False if already present), `update_video(dataset_id, video_id, **fields)`, `list_videos(dataset_id) -> list[dict]`, `mark_stage(dataset_id, video_id, stage, done: bool)`, `stage_done(dataset_id, video_id, stage) -> bool`, `replace_clips(dataset_id, video_id, clips: list[dict])`, `update_clip(dataset_id, clip_id, **fields)`, `clips(dataset_id, *, video_id=None, statuses=None) -> list[dict]`, `get_thresholds() -> Thresholds`, `set_thresholds(Thresholds)`. `STAGES: tuple[Stage, ...]` in `pipeline/__init__.py`.

- [ ] **Step 1: Write failing tests**: `test_add_video_dedupes` (second `add_video` with the same id returns False, `list_videos` has one row), `test_stage_marks_roundtrip`, `test_replace_clips_is_idempotent`, `test_clips_filter_by_status`, `test_thresholds_default_then_saved`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement `store.py`.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(ytds): sqlite store`.

### Task 3: Event bus

**Files:**
- Create: `ytds/events.py`
- Test: `tests/test_events.py`

**Interfaces:**
- Produces: `EventBus(min_interval_s: float = 0.25, clock=time.monotonic)`; `publish(event: dict) -> None` (thread-safe; called from worker threads); `subscribe(dataset_id: str) -> asyncio.Queue` and `unsubscribe(dataset_id, queue)`; `bind_loop(loop)` (publishes via `loop.call_soon_threadsafe`). Event shape per spec §4.5, keyed by `dataset_id` (the spec's `job_id` is the dataset id). Types: `progress`, `stage_done`, `stage_failed`, `video_done`, `log`, `llm_status`.
- Throttle: `progress` events for the same `(dataset_id, video_id, stage)` within `min_interval_s` are dropped, except when `done == total`. Other types are never dropped.

- [ ] **Step 1: Failing tests** with a fake clock: `test_progress_throttled_to_4_per_second`, `test_final_progress_always_sent`, `test_stage_events_never_throttled`, `test_subscribers_only_get_their_dataset`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(ytds): event bus with progress throttle`.

### Task 4: Download stage

**Files:**
- Create: `ytds/pipeline/download.py`
- Test: `tests/test_download.py`

**Interfaces:**
- Produces:
  - `expand_urls(text: str, extract=None) -> list[tuple[str, str]]`: (video_id, url) pairs, playlists expanded with yt-dlp `extract_flat="in_playlist"`, order kept, duplicates removed. `extract` is injectable (default `yt_dlp.YoutubeDL(...).extract_info`).
  - `download(video_id: str, url: str, out_dir: Path, on_progress: Callable[[int,int],None], cookies: Path | None = None, ydl_factory=None) -> dict` returns `{title, channel, upload_date, duration_s, license}` and writes `out_dir/audio.wav` (48 kHz mono, via ffmpeg postprocessor `FFmpegExtractAudio` to wav then `ffmpeg -ac 1 -ar 48000`). Progress maps yt-dlp `downloaded_bytes`/`total_bytes` (or `total_bytes_estimate`). Format `bestaudio/best`. `retries=5`, `fragment_retries=5`.
  - `DownloadError(RuntimeError)` holding yt-dlp's last error line.

- [ ] **Step 1: Failing tests** with fake extract/ydl: `test_expand_dedupes_same_video_twice`, `test_expand_playlist_and_member_url_dedupe` (Review Focus 1), `test_progress_callback_gets_bytes`, `test_metadata_fields_mapped`, `test_error_line_kept` (fake raises `yt_dlp.utils.DownloadError("ERROR: [youtube] x: Private video")` → `DownloadError` message contains `Private video`). Plus `@pytest.mark.network test_real_download_short_video` using one short public Bangla video URL put in `tests/data/smoke_url.txt` (implementer picks a Creative Commons one and records it).
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS (network test skipped).
- [ ] **Step 5: Commit** `feat(ytds): yt-dlp download stage`.

### Task 5: GPU worker and separation stage

**Files:**
- Create: `ytds/gpu.py`, `ytds/pipeline/separate.py`
- Test: `tests/test_gpu.py`, `tests/test_separate.py`

**Interfaces:**
- Produces:
  - `GpuWorker(loaders: dict[str, Callable[[], Any]], empty_cache=torch_empty_cache)`. `use(name) -> model` loads `name` if it is not the loaded one, first dropping the previous model and calling `empty_cache()`. `release()`. `loaded: str | None`.
  - `run_with_oom_retry(fn: Callable[[float], T], first: float) -> T`: calls `fn(first)`; on `torch.cuda.OutOfMemoryError` (or `RuntimeError` containing `"out of memory"`) calls `empty_cache` and retries once with `first / 2`; a second OOM propagates.
  - `separate(audio_wav: Path, out_wav: Path, model, segment_s: float, on_progress) -> None`: demucs `apply_model(model, wav, segment=segment_s, split=True, overlap=0.25, progress=False)`; writes the `vocals` stem as mono 44.1 kHz WAV. Progress reported per chunk (wrap in a manual chunk loop of 60 s blocks so progress is visible).
  - Loader name `"demucs"` → `demucs.pretrained.get_model("htdemucs")` on cuda, eval.

- [ ] **Step 1: Failing tests**: `test_use_swaps_and_empties_cache`, `test_use_same_model_no_reload`, `test_oom_retry_halves_once`, `test_second_oom_raises`, `test_separate_writes_mono_vocals` (fake model returning a 4-stem tensor; asserts the output is mono and equals the stem index for `vocals`).
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(ytds): gpu worker and demucs separation`.

### Task 6: Segment and quality stages

**Files:**
- Create: `ytds/pipeline/segment.py`, `ytds/pipeline/quality.py`
- Test: `tests/test_segment.py`, `tests/test_quality.py`

**Interfaces:**
- Produces:
  - `plan_clips(regions: list[tuple[float,float]], *, merge_gap_s, min_s, max_s) -> list[tuple[float,float]]`. Pure. Algorithm: merge regions whose gap < `merge_gap_s`; a merged span longer than `max_s` is split at its longest internal gap (recursively, using the original region boundaries as cut points; a span with no internal gap is cut hard at `max_s`); a span shorter than `min_s` merges with the nearer neighbour if the result is ≤ `max_s`, else it is dropped.
  - `vad_regions(vocals_16k: np.ndarray) -> list[tuple[float,float]]` via `silero_vad.get_speech_timestamps(..., return_seconds=True)`.
  - `segment(vocals_wav: Path, th: Thresholds) -> list[dict]` → clip dicts `{idx, start_s, end_s}`.
  - `speaker_filter(embeddings: np.ndarray, dnsmos: np.ndarray, min_sim: float) -> tuple[np.ndarray, np.ndarray]` → (similarities, keep mask). Centroid = L2-normalized median of the embeddings of clips with DNSMOS ≥ the median.
  - `quality(vocals_wav: Path, clips: list[dict], th, scorer=None, embedder=None) -> list[dict]` sets `dnsmos` (OVRL from `score_dnsmos`), `speaker_sim`, and `drop_reason` (`dnsmos` first, then `other_speaker`), status `dropped` for dropped clips.

- [ ] **Step 1: Failing tests**: `test_merges_short_gaps`, `test_splits_long_span_at_longest_gap`, `test_hard_cut_without_gap`, `test_short_tail_merges_with_neighbour`, `test_isolated_short_dropped`, `test_all_clips_within_limits` (property-style over 200 random region lists: every clip within [min_s, max_s]), `test_no_regions_gives_no_clips`; `test_speaker_filter_drops_outlier` (9 similar vectors + 1 orthogonal), `test_quality_sets_drop_reasons` with fake scorer/embedder.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(ytds): VAD segmenting and quality gate`.

### Task 7: Transcription and routing

**Files:**
- Create: `ytds/pipeline/transcribe.py`, `ytds/pipeline/routing.py`
- Test: `tests/test_routing.py`, `tests/test_transcribe.py`

**Interfaces:**
- Produces:
  - `cer(ref: str, hyp: str) -> float`: character edit distance over `bn_normalize(..., mode="asr")` with spaces removed, divided by `max(len(ref), 1)`; both empty → 0.0.
  - `route(text_a: str, text_b: str, th: Thresholds) -> tuple[ClipStatus, float]`: A empty (after normalize) → (`"fallback"`, 1.0); cer ≤ `cer_auto` → `"auto"`; ≤ `cer_fallback` → `"flagged"`; else `"fallback"`. Note `"fallback"` is a routing result; the runner stores it as status `pending_llm` until Task 8's stage runs.
  - `load_model_a() -> Transcriber`, `load_model_b() -> Transcriber`, where `Transcriber = Callable[[list[np.ndarray]], list[str]]` (16 kHz mono input). Model A ports `prompt_text`, `generation_inputs`, `max_new_tokens`, `split_transcript` from `backend/workers/backends/qwen-asr/mlops_backend/` (copy with a source comment; that venv is Linux-only), loads `Qwen3ASRForConditionalGeneration` in bfloat16 with `attn_implementation="sdpa"`, batch 8. Model B: faster-whisper `WhisperModel(<ct2 dir>, device="cuda", compute_type="int8_float16")`, `language="bn"`, `beam_size=1`; the CT2 conversion of `mozilla-ai/whisper-large-v3-turbo-bn` is cached under `var/models/` (reuse the conversion code in `stt-playground/stt_ui.py`).
  - `transcribe_clips(clips, vocals_wav, gpu: GpuWorker, on_progress) -> None`: stage-major — all clips through A, then `gpu.use("model_b")`, all through B; sets `text_a`, `text_b`, `cer`, `status`, `text_source="stt"`.

- [ ] **Step 1: Failing tests**: `test_cer_identical_is_zero`, `test_cer_both_empty_is_zero`, `test_route_boundaries` (CER exactly 0.05 → auto, exactly 0.30 → flagged), `test_route_empty_a_is_fallback` (Review Focus 3), `test_route_latin_and_digits_no_crash` (`"iPhone 15 এর দাম"` vs `"আইফোন ১৫ এর দাম"`), `test_transcribe_clips_stage_major` (fake GpuWorker records the `use()` order `["model_a","model_b"]` for 3 clips). `@pytest.mark.gpu test_model_a_loads_and_transcribes` on one Bangla clip from `tests/data/`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** If `Qwen3ASRForConditionalGeneration` does not import on Windows, stop and report (spec §10 fallback order needs the user's go).
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(ytds): dual-model transcription and CER routing`.

### Task 8: LLM client, punctuation guard, fallback stage

**Files:**
- Create: `ytds/llm.py`, `ytds/pipeline/punct.py`, `ytds/pipeline/fallback.py`
- Test: `tests/test_llm.py`, `tests/test_punct.py`, `tests/test_fallback.py`

**Interfaces:**
- Produces:
  - `OllamaClient(host: str, model: str, api_key: str | None, transport: httpx.BaseTransport | None = None)`. `show() -> bool` (POST `/api/show`). `chat(system: str, user: str) -> str` (POST `/api/chat`, `stream=false`, `options.temperature=0`; header `Authorization: Bearer <key>` only when `api_key`). Retries 3 times with backoff 1, 2, 4 s on connect errors, 429 and 5xx (sleep injectable). Raises `LlmUnavailable` after that.
  - `words_unchanged(original: str, punctuated: str) -> bool`: strip `।?!,;:"'“”‘’()-—` and `.`, then compare `bn_normalize(..., mode="asr").split()`.
  - `punctuate(text: str, llm) -> tuple[str, bool]` → (text, no_punct). Rejected output or `LlmUnavailable` → (original, True).
  - `translate_en_to_bn(en: str, llm) -> str`.
  - `run_fallback(clips, vocals_wav, gpu, llm, on_progress)`: `gpu.use("whisper_translate")` (faster-whisper `large-v3`, `task="translate"`, `language="bn"`), sets `text_en`; then LLM sets `text_translated`, `text_final`, `text_source="translated"`, status `needs_review`. On `LlmUnavailable` the clip stays `pending_llm`.
  - Prompts (exact copy, in `llm.py`): punctuation system prompt `"You add punctuation to Bangla transcripts. Return the same words in the same order. Only add or change punctuation marks (। ? , ! and quotes). Do not fix spelling, do not translate, do not add or remove words. Output only the text."`; translation system prompt `"Translate the English sentence into natural spoken Bangla, as a Bangladeshi speaker would say it. Keep English product and brand names in Latin script. Output only the Bangla text."`

- [ ] **Step 1: Verify the slug.** Check ollama.com for the Gemma 4 cloud tag. If it is not `gemma4:31b-cloud`, update the default in `config.py`, `.env.example` and the spec §1.1/§2.1, and note it in the commit.
- [ ] **Step 2: Failing tests** with `httpx.MockTransport`: `test_bearer_only_with_key`, `test_retries_then_unavailable` (3 × 503 → `LlmUnavailable`, sleeps `[1,2,4]`), `test_show_false_on_404`; `test_guard_accepts_punct_only`, `test_guard_rejects_changed_word`, `test_guard_rejects_dropped_word`, `test_punctuate_falls_back_on_unavailable`; `test_fallback_sets_translated_needs_review`, `test_fallback_keeps_pending_llm_when_llm_down`.
- [ ] **Step 3: Run.** Expected: FAIL.
- [ ] **Step 4: Implement.**
- [ ] **Step 5: Run.** Expected: PASS.
- [ ] **Step 6: Commit** `feat(ytds): ollama client, punctuation guard, translate fallback`.

### Task 9: Job runner

**Files:**
- Create: `ytds/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `Store`, `EventBus`, `GpuWorker`, stage functions from Tasks 4–8.
- Produces: `JobRunner(store, bus, gpu, llm, settings, stages: StageFns | None = None)` where `StageFns` is a dataclass of the seven stage callables (injectable for tests). `start() / stop()`; `enqueue(dataset_id)`; `retry(dataset_id, video_id)` (clears the failed stage's mark, sets status `queued`); `skip(dataset_id, video_id)`; `retry_llm(dataset_id)`.
- Behaviour:
  - A thread pool (2 workers) runs `download`, then `segment` and `quality`'s CPU part, per video.
  - One GPU thread loops: picks the next GPU stage that has ready videos in `STAGES` order (`separate`, then `quality`'s embedder, `transcribe`, `fallback`) and runs it for all ready videos before switching models.
  - `punctuate` runs on the pool after `transcribe` for `auto`/`flagged` clips.
  - A stage is skipped when `store.stage_done` is true (resume). Any exception marks the video `failed` with `str(e)` and publishes `stage_failed`; other videos continue.
  - After the last stage: if kept duration < 30 s, set `warning="<30 s clean audio"`; if > 50 % of clips went to fallback, set `warning="poor STT"`; status `done`; publish `video_done`.
  - On `start()`, videos with status `running` are reset to `queued` (restart resume).

- [ ] **Step 1: Failing tests** with fake stages and a temp store: `test_runs_all_stages_in_order`, `test_resume_skips_done_stages` (pre-mark `download`,`separate` done; fakes record calls), `test_failure_isolated_to_video`, `test_gpu_stage_major_across_videos` (2 videos: `separate` runs for both before `transcribe` for either), `test_no_speech_video_done_with_warning` (Review Focus 2: segment returns `[]`), `test_running_reset_on_start` (Review Focus 4), `test_poor_stt_warning`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(ytds): job runner with resume and stage-major GPU scheduling`.

### Task 10: Splits and export

**Files:**
- Create: `ytds/splits.py`, `ytds/exporter.py`
- Test: `tests/test_splits.py`, `tests/test_exporter.py`

**Interfaces:**
- Produces:
  - `assign_splits(clips: list[dict], seed: int = 0, ratios=(0.90, 0.05, 0.05), min_video_s: float = 60.0) -> dict[str, str]` (clip id → split). Per video, clips in `idx` order: a video with < `min_video_s` kept seconds → all `train`. Otherwise val and test are each one contiguous run with duration ≈ ratio × video duration (at least one clip each), start positions drawn from `random.Random(f"{seed}:{video_id}")`, runs not overlapping.
  - `exportable(store, dataset_id, include_unreviewed_translated: bool) -> tuple[list[dict], int]` → (clips, unreviewed_translated_count): statuses `auto`, `approved`; plus `flagged` is NOT exportable until approved.
  - `export_dataset(store, dataset_id, out_dir: Path, on_progress, *, include_unreviewed_translated=False) -> dict` (summary). Raises `UnreviewedTranslated(count)` when count > 0 and the flag is false. Per clip: read the vocals stem slice, call `dataprep.audio_proc.process_tts(y, sr, segments=[(0, dur)])`, write `wavs/<clip_id>.wav` (`soundfile`, `PCM_16`). Then `metadata.csv` (UTF-8, `file,text,speaker,split`), `curation.json` (spec §7, including `commercial_use: false`, git commit via `git rev-parse HEAD`). Finally run `app.datasets.validate_audio.validate_audio(out_dir, "tts")`; failing clips are removed from `metadata.csv` and counted in `curation.json` as `validation_dropped`. Writes go to `out_dir.with_suffix(".tmp")` and rename at the end.

- [ ] **Step 1: Failing tests**: `test_ratios_within_tolerance` (10 videos × 300 s of 5 s clips: val and test each 5 % ± 1 % of total), `test_val_test_contiguous`, `test_deterministic_for_seed`, `test_short_video_all_train`; `test_export_layout` (fake store, 3 synthetic clips of tone + noise: `metadata.csv` header exactly `file,text,speaker,split`, wavs are 24000 Hz mono PCM_16), `test_refuses_unreviewed_translated`, `test_flagged_unapproved_excluded`, `test_curation_has_provenance_and_commercial_false`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(ytds): deterministic splits and platform-layout export`.

### Task 11: HTTP API and WebSocket

**Files:**
- Create: `ytds/api/__init__.py`, `api/jobs.py`, `api/review.py`, `api/dataset.py`, `api/settings.py`, `api/ws.py`, `ytds/__main__.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: Tasks 2, 3, 8, 9, 10.
- Produces: `create_app(settings, runner=None, store=None, bus=None) -> FastAPI`; `python -m ytds` runs uvicorn on `settings.port`. Routes (JSON):
  - `POST /api/datasets {name, speaker?, urls}` → `{dataset_id, added, duplicates}`; `GET /api/datasets`; `GET /api/datasets/{id}` → dataset + videos (with per-stage done flags, counts kept/dropped, warning, error).
  - `POST /api/datasets/{id}/videos {urls}`; `POST /api/datasets/{id}/videos/{vid}/retry|skip`; `DELETE /api/datasets/{id}/videos/{vid}`.
  - `GET /api/datasets/{id}/review?reason=&video=&status=` → clips with status `flagged`, `needs_review`, `pending_llm`; `PATCH /api/datasets/{id}/clips/{clip_id} {text_final?, status?: "approved"|"rejected"}`; `GET /api/datasets/{id}/clips/{clip_id}/audio` → WAV slice of the vocals stem, written once to `var/videos/<vid>/clips/<clip_id>.wav` and served with `FileResponse` (Range support).
  - `GET /api/datasets/{id}/stats` → hours kept, drops per reason, per-video breakdown, split preview from `assign_splits`; `POST /api/datasets/{id}/export {out_dir?, include_unreviewed_translated?}` → 409 `{unreviewed: n}` or 202; export progress goes over the WebSocket with `stage="export"`.
  - `GET/PUT /api/settings` (thresholds, speaker, cookies path, llm host/model); `POST /api/settings/llm-test` → `{ok, error?}`; `GET /api/health` → `{llm_ok, cuda, ffmpeg}`.
  - `WS /ws/datasets/{id}` streams bus events.
  - Serves `frontend/dist` at `/` when it exists.

- [ ] **Step 1: Failing tests** with `TestClient`, fake runner and fake bus: `test_create_dataset_dedupes_urls`, `test_review_patch_approve_and_edit`, `test_export_409_with_unreviewed`, `test_ws_receives_published_event`, `test_clip_audio_supports_range` (Review Focus 5: header `Range: bytes=0-99` → 206 and 100 bytes), `test_health_shape`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -v`. Expected: all PASS.
- [ ] **Step 5: Commit** `feat(ytds): REST API, review endpoints and job websocket`.

### Task 12: Frontend scaffold, Jobs page, live progress

**Files:**
- Create: `frontend/package.json`, `vite.config.ts`, `tsconfig.json`, `index.html`, `src/main.tsx`, `src/App.tsx`, `src/api.ts`, `src/types.ts`, `src/useJobEvents.ts`, `src/theme.css`, `src/pages/Jobs.tsx`, `src/components/ProgressRow.tsx`
- Test: `src/__tests__/ProgressRow.test.tsx`, `src/__tests__/useJobEvents.test.ts`

**Interfaces:**
- Consumes: Task 11 routes and event shapes.
- Produces: `types.ts` mirrors the shared types; `api.ts` one typed function per route; `useJobEvents(datasetId): {videos: VideoState[], connected: boolean, log: string[]}` — fetches `GET /api/datasets/{id}` on connect and on every reconnect (backoff 1, 2, 4, max 10 s), then applies events; `ProgressRow({video})` renders title, current stage, one bar per stage, kept/dropped, ETA, status, warning and the retry/skip/delete buttons.
- Vite dev proxy: `/api` and `/ws` → `http://localhost:8765`.

- [ ] **Step 1: Load the `frontend-design` and `ui-ux-pro-max` skills** and set the visual direction: dense tool UI, left nav (Jobs, Review, Dataset, Settings), light and dark themes as CSS tokens in `theme.css`, Bangla-capable font stack (`"Noto Sans Bengali", system-ui`).
- [ ] **Step 2: Scaffold** with React 18, TypeScript strict, Vitest + jsdom + Testing Library; `npm install`.
- [ ] **Step 3: Failing tests**: `ProgressRow shows download percent from progress event`, `ProgressRow shows retry only when failed`, `useJobEvents refetches state after reconnect` (Review Focus 4; mock WebSocket closes then reopens; fetch called twice), `useJobEvents applies stage_done`.
- [ ] **Step 4: Run `npm test`.** Expected: FAIL.
- [ ] **Step 5: Implement** Jobs page (URL textarea, dataset name, Start; dataset picker; overall bar; table of `ProgressRow`; collapsible log; red LLM banner from `/api/health`).
- [ ] **Step 6: Run `npm test` and `npx tsc --noEmit`.** Expected: PASS, no type errors.
- [ ] **Step 7: Commit** `feat(ytds-ui): jobs page with live progress`.

### Task 13: Review, Dataset and Settings pages

**Files:**
- Create: `src/pages/Review.tsx`, `src/pages/Dataset.tsx`, `src/pages/Settings.tsx`, `src/components/ReviewCard.tsx`, `src/components/Waveform.tsx`
- Test: `src/__tests__/ReviewCard.test.tsx`

**Interfaces:**
- Consumes: Task 11 review, stats, export and settings routes; Task 12 `api.ts`.
- Produces: `ReviewCard({clip, onApprove, onReject, onNext, onPrev, onEdit})`. Keys (ignored while the text box has focus, except `Enter` with Ctrl): `Space` play/pause, `Enter` approve, `R` reject, `J` next, `K` previous. Text edits are saved with a 500 ms debounce via `PATCH`. A word-level diff highlights A vs B (a small LCS over whitespace tokens; no dependency). `Waveform` draws peaks from the decoded clip on a canvas.

- [ ] **Step 1: Failing tests**: `Enter approves`, `R rejects`, `J and K navigate`, `keys ignored while typing in the text box`, `edit is saved after debounce`, `translated clip shows English and translation`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** the Review page (filters, counter reviewed/remaining), the Dataset page (stats, per-video table, split preview, Export with path field, 409 dialog offering "export without them", export progress bar from the WebSocket) and the Settings page (spec §4.4 fields, LLM Test button).
- [ ] **Step 4: Run `npm test` and `npx tsc --noEmit`.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(ytds-ui): review queue, dataset export and settings`.

### Task 14: Launcher, README and GPU smoke test

**Files:**
- Create: `youtube-tts-dataset/run.ps1`, `youtube-tts-dataset/README.md`, `backend/tests/test_smoke_gpu.py`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: `run.ps1`** (PowerShell 5.1 syntax): checks `ffmpeg -version` and `nvidia-smi` (print the fix and exit 1 if missing: `winget install Gyan.FFmpeg`); `uv sync` in `backend`; `npm install` and `npm run build` in `frontend` when `dist` is missing or `-Rebuild` is passed; starts `uv run python -m ytds`; prints `http://localhost:<port>`. A `-Dev` switch starts the backend and `npm run dev` in two windows instead.
- [ ] **Step 2: `test_smoke_gpu.py`** marked `gpu` and `network`: runs the real runner on the URL in `tests/data/smoke_url.txt`, approves every review clip through the store, exports to `tmp_path`, asserts `metadata.csv` has ≥ 1 row, `validate_audio` reports no errors, and `torch.cuda.max_memory_allocated() < 7 * 1024**3`.
- [ ] **Step 3: Run** `uv run pytest -m "gpu and network" -v` on the RTX 4060. Expected: PASS. Record peak VRAM and wall time in the README.
- [ ] **Step 4: README**: what it does, setup (ffmpeg, uv, Node 20+, `ollama signin`), `.\run.ps1`, the pipeline table, thresholds, output layout, how to ingest via the platform's "From server path", and the R&D-only licence note.
- [ ] **Step 5: Manual check** with the `run` skill: start the app, add 2 videos, watch progress bars move, review 3 clips by keyboard, export. Fix what breaks.
- [ ] **Step 6: Commit** `feat(ytds): launcher, README and GPU smoke test`.
