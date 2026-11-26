# YouTube TTS dataset builder — design

Date: 2026-11-26. Status: approved in brainstorming, awaiting spec review.

## 1. Goal

Turn a shortlist of YouTube videos of one Bangla speaker into a stage-B TTS voice dataset in the platform's `tts` layout (speech dataprep spec §4.4, §8.3, §9). The tool downloads audio, removes music, cuts clips, transcribes them in Bangla, lets the user review doubtful clips, and exports `metadata.csv` + `wavs/` + `curation.json`.

It is a standalone sub-project in this repo with its own web UI. It shows live progress for downloads and every processing stage.

### 1.1 Decisions from brainstorming

| Topic | Decision |
|---|---|
| Speakers | One speaker per dataset (stage B). No diarization. Clips with another voice are dropped. |
| Background music | Some videos have it. Every video goes through vocal separation. |
| Transcription | Bangla STT first. An English STT + LLM translation path is the fallback for clips the Bangla models fail on. |
| Translated clips | Tagged `text_source=translated`. They always go to the review queue and are not exported until approved or edited. |
| LLM | Ollama cloud model (default `gemma4:31b-cloud`; slug verified against ollama.com during implementation). Used for punctuation (words locked) and for the fallback translation. |
| Review | Clips the two Bangla STT models agree on pass automatically. Flagged and translated clips are reviewed in the UI. |
| Stack | FastAPI backend + Vite/React/TypeScript frontend. |
| Machine | User's Windows PC, RTX 4060 8 GB, Windows PowerShell 5.1. Not the VM. |
| Splits | 90/5/5 train/val/test by duration (§6). |

### 1.2 Out of scope

- Multi-speaker sources and diarization (stage A data comes from `tools/dataprep`).
- Training or evaluating TTS models.
- Ingesting into the platform. The export lands in a folder the platform's "From server path" upload reads.
- Copyright clearance. The tool records provenance (§7) so YouTube-derived data can be filtered out before any commercial use. YouTube content is assumed to be R&D-only unless the user records a licence.

## 2. Layout and runtime

```
youtube-tts-dataset/
  backend/            FastAPI app, pipeline stages, SQLite models (uv project)
    ytds/
      api/            REST routes + WebSocket
      pipeline/       one module per stage (§3)
      gpu.py          single GPU worker, model load/unload
      llm.py          Ollama client
      store.py        SQLite access
      config.py       settings + .env lookup
    tests/
  frontend/           Vite + React + TypeScript
  run.ps1             starts backend and frontend (PowerShell 5.1 syntax)
  .env.example
  README.md
var/ (git-ignored, under youtube-tts-dataset/)
  ytds.db
  videos/<video_id>/<stage>/...
```

- Python 3.12, uv, torch with CUDA 12.x wheels for Windows. No flash-attn; attention uses SDPA.
- ffmpeg is required on PATH (`winget install Gyan.FFmpeg`). `run.ps1` checks for it and for `nvidia-smi`.
- The backend serves the built frontend in production mode. In dev, Vite proxies `/api` and `/ws` to the backend.

### 2.1 Configuration

Settings are read from, in order: environment variables, `youtube-tts-dataset/.env`, the repo root `.env`. The first value found wins.

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Local daemon. When signed in (`ollama signin`), it routes `*-cloud` models to Ollama cloud. |
| `OLLAMA_API_KEY` | empty | Needed only when `OLLAMA_HOST=https://ollama.com` (direct cloud API, bearer auth). |
| `LLM_MODEL` | `gemma4:31b-cloud` | Model for punctuation and translation. |
| `YTDS_DATA_DIR` | `youtube-tts-dataset/var` | Work data. |
| `YTDS_EXPORT_DIR` | `<repo>/var/import` | Default export root. |
| `YTDS_PORT` | `8765` | Backend port. |
| `HF_TOKEN` | empty | Only if a model download needs it. |

Thresholds (§3.1) are stored in SQLite and edited on the Settings page. Their defaults live in `config.py`.

At startup the backend calls the LLM host's `/api/show` for `LLM_MODEL`. If that fails, the UI shows a red banner. Pipeline stages that do not need the LLM still run.

## 3. Pipeline

Each video moves through the stages below. A stage writes its outputs under `var/videos/<video_id>/<stage>/` and marks itself done in SQLite only after all outputs are written. On restart, stages not marked done run again from the start; done stages are skipped.

| # | Stage | Tool | Device |
|---|---|---|---|
| 1 | `download` | yt-dlp, best audio only; ffmpeg converts to 48 kHz mono WAV. Saves title, channel, upload date, URL, duration, licence field. A playlist URL expands into its videos. | network |
| 2 | `separate` | Demucs `htdemucs`, vocals stem, chunked (`segment` setting). | GPU, about 3 GB |
| 3 | `segment` | Silero VAD on the vocals stem. Merge pauses shorter than 300 ms. Clips must be 2–15 s; longer regions are split at the longest internal pause; shorter ones are merged with a neighbour or dropped. | CPU |
| 4 | `quality` | DNSMOS P.835 (onnx) per clip: drop below `dnsmos_min`. ECAPA-TDNN embedding per clip; the video's speaker centroid is the median of the top-DNSMOS half; drop clips with cosine similarity below `speaker_sim_min`. | CPU / GPU |
| 5 | `transcribe` | Model A: `SayedShaun/Qwen3-ASR-0.6B-Bengali` (transformers). Model B: whisper-large-v3-turbo-bn (faster-whisper, int8_float16). CER between A and B after `bn_normalize(mode="asr")` routes the clip (§3.1). | GPU, about 2 GB each, one at a time |
| 6 | `fallback` | Whisper large-v3, `task=translate`, gives English. The LLM translates the English to Bangla. Text tagged `translated`, status `needs_review`. | GPU + LLM |
| 7 | `punctuate` | The LLM adds punctuation (। ? , ! and quotes) to model A's text. Guard in §3.2. | LLM |
| 8 | `review` | Manual, in the UI (§4.2). | — |
| 9 | `export` | Per dataset, not per video (§5). | CPU |

Stages 1–7 run automatically after a job starts. Stage 9 starts when the user clicks Export.

### 3.1 Routing after transcription

| Condition | Status | Text used |
|---|---|---|
| CER(A, B) ≤ `cer_auto` (default 5 %) | `auto` | A, punctuated |
| `cer_auto` < CER ≤ `cer_fallback` (default 30 %) | `flagged`, goes to review | A, punctuated; B shown alongside |
| CER > `cer_fallback`, or A is empty | `fallback`, then `needs_review` | translated text; A and B shown alongside |

If more than 50 % of a video's clips route to `fallback`, the video gets a "poor STT" warning in the Jobs table.

### 3.2 Punctuation guard

The LLM is asked to add punctuation only. The backend strips all punctuation from the LLM output, normalizes both texts with `bn_normalize(mode="asr")`, and compares the word sequences. If they differ in any way, the output is rejected and the clip keeps its unpunctuated text with the flag `no_punct`.

### 3.3 GPU scheduling

There is one GPU worker. It holds at most one model at a time. It works stage by stage across all queued videos (for example, separate every downloaded video, then transcribe with model A across all clips, then model B), so each model is loaded once per batch, not once per video. Between stages it unloads the model and calls `torch.cuda.empty_cache()`. Downloads (network) and CPU stages run in a separate thread pool, so downloading continues while the GPU works.

## 4. UI

The UI has four pages in a left navigation bar. Visual design is produced at implementation time with the `frontend-design` and `ui-ux-pro-max` skills. It is a dense tool interface with light and dark themes.

### 4.1 Jobs

- A form with a text area for YouTube URLs (one per line, playlists allowed), a dataset name, and Start.
- A table with one row per video: title, duration, current stage, a progress bar per stage (download % from yt-dlp, clips n/N for STT), clips kept and dropped, ETA, and status.
- An overall progress bar for the job.
- A collapsible live log.
- Per-video actions: retry the failed stage, skip, delete.

### 4.2 Review

- A queue of `flagged` and `needs_review` clips, filterable by reason, video and status.
- Each clip card shows an audio player with a waveform, text A and text B with differences highlighted, the English text and the LLM translation for translated clips, and an editable final-text box.
- Keyboard: `Space` play/pause, `Enter` approve, `R` reject, `J` next, `K` previous. Edits are saved automatically.
- A counter: reviewed / remaining.

### 4.3 Dataset

- Hours and clips kept; clips dropped per reason (DNSMOS, other speaker, length, rejected, unreviewed).
- A per-video breakdown.
- A split preview (hours and clips per split).
- Export button with an output path (default `<YTDS_EXPORT_DIR>/<dataset>/`) and its own progress bar.

### 4.4 Settings

- Thresholds: `cer_auto`, `cer_fallback`, `dnsmos_min`, `speaker_sim_min`, clip min/max length, Demucs segment size.
- LLM host, model, and a "Test connection" button.
- Speaker name written to `metadata.csv` (default: the dataset name).
- Optional path to a `cookies.txt` for age-gated videos.
- The model used for each stage (read-only list).

### 4.5 Live progress

The backend pushes events over a WebSocket at `/ws/jobs/<job_id>`. Each event is JSON:

```json
{"type": "progress", "job_id": "...", "video_id": "...", "stage": "transcribe",
 "done": 120, "total": 640, "eta_s": 310, "message": "model A"}
```

Other types are `stage_done`, `stage_failed` (with `error`), `video_done`, `log`, and `llm_status`. On (re)connect the UI fetches the full job state over REST, then applies events. Progress events are throttled to at most 4 per second per video.

## 5. Export

1. Collect clips with status `auto`, approved `flagged`, and approved `needs_review` (translated) clips. If any translated clip is still unreviewed, Export refuses with a count. The user can then choose to export without those clips.
2. Assign splits (§6).
3. Process each clip, reusing the `tools/dataprep` audio code (dataprep spec §8.3): trim leading and trailing non-speech with 150 ms padding, loudness-normalize to −23 LUFS with a −1 dBTP true-peak limit, resample to 24 kHz, write 16-bit PCM WAV. The source is the separated vocals stem.
4. Check every clip against the platform's `validate_audio` rules (sample rate, mono, LUFS window, length). A failing clip is skipped and counted.
5. Write `metadata.csv` with columns `file,text,speaker,split`, where `file` is `wavs/<name>.wav`. Text is the final punctuated Bangla text.
6. Write `curation.json` (§7).

## 6. Splits

- Ratios are 90/5/5 train/val/test, measured by audio duration.
- Each video contributes to all three splits in proportion to its kept duration. Val and test take contiguous runs of clips from each video, not scattered clips, so neighbouring sentences do not leak between splits.
- A fixed seed picks where in each video the runs start, so a re-export gives the same splits.
- A video with less than 60 s of kept audio goes entirely to train.
- The test split is held out for TTS evaluation (CER of a fixed STT model on synthesized audio).

## 7. Provenance (`curation.json`)

- Tool name and git commit, export time, thresholds and model IDs used.
- Per video: URL, title, channel, upload date, licence field from yt-dlp, clips and seconds kept, and seconds dropped per reason.
- Counts: total clips, hours per split, clips per `text_source` (`stt` or `translated`), clips with `no_punct`.
- A `commercial_use` field, default `false`, for the user to change only when a licence is recorded.

## 8. Error handling

| Failure | Behaviour |
|---|---|
| Download fails (private, removed, geo-blocked, age-gated) | Video marked `failed` with yt-dlp's last error line. Other videos continue. |
| GPU out of memory | The stage retries once with a smaller chunk or batch. A second failure marks the video `failed` with the error. |
| LLM unreachable or rate-limited | 3 retries with exponential backoff. Punctuation then falls back to the raw text with `no_punct`. Fallback clips stay in `pending_llm`; the "Retry LLM" button reruns them. |
| Punctuation changes words | Output rejected (§3.2). |
| Video yields less than 30 s of clean audio | Warning on the video row, not an error. |
| Backend restarted mid-job | Unfinished stages rerun on start; done stages are skipped. |
| ffmpeg or CUDA missing | `run.ps1` stops with the fix to apply. |

## 9. Testing

pytest. Tests marked `gpu` or `network` are skipped by default.

- Unit tests:
  - Segmenter on synthetic audio: silence and tone patterns give the expected cut points and length limits.
  - CER routing at the threshold boundaries.
  - Punctuation guard: punctuation-only changes accepted, any word change rejected.
  - Split assignment: ratios within tolerance, contiguous runs, deterministic for a seed, short videos go to train.
  - `metadata.csv` and `curation.json` layout.
  - `.env` lookup order.
- API tests with fake stage runners: create a job, the WebSocket event shapes, and the review edit, approve and reject flow.
- Heavy smoke test (`-m "gpu and network"`) on the RTX 4060: one short public Bangla clip goes through the full pipeline to export. Peak VRAM must stay under 7 GB.
- Frontend: `tsc --noEmit`, plus component tests for the progress row and the review card keyboard handling (Vitest + Testing Library).

## 10. Risks and open items

- `SayedShaun/Qwen3-ASR-0.6B-Bengali` on Windows: the `qwen-asr` package is tested on Linux only. If it does not install on Windows, model A loads through plain transformers. If that also fails, model A becomes IndicConformer-bn through NeMo, and the plan records the change.
- The Gemma 4 cloud slug is checked on ollama.com during implementation.
- Demucs on 8 GB: `segment` defaults to 7 s; the OOM retry halves it.
- YouTube rate limits on large playlists: downloads run 2 at a time with yt-dlp's own retry options.
