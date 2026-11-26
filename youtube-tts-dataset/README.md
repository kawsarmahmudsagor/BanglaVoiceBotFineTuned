# YouTube TTS dataset builder (`ytds`)

Turns a shortlist of YouTube videos of **one Bangla speaker** into a TTS voice dataset in the platform's
`tts` layout (`wavs/` + `metadata.csv` + `curation.json`). It downloads the audio, removes music, cuts
2–15 s clips, drops noisy clips and other voices, transcribes with two Bangla STT models, lets you review
doubtful clips in a web UI, and exports. Progress for every stage is shown live.

Runs on a Windows 11 PC with an NVIDIA GPU (built and tested on an RTX 4060 8 GB), Windows PowerShell 5.1.
Design: [`docs/superpowers/specs/2026-11-26-youtube-tts-dataset-design.md`](../docs/superpowers/specs/2026-11-26-youtube-tts-dataset-design.md).

## Setup

| Need | Install |
|---|---|
| ffmpeg on PATH | `winget install Gyan.FFmpeg` |
| NVIDIA driver (`nvidia-smi` works) | from nvidia.com |
| uv | `winget install astral-sh.uv` |
| Node.js 20+ and npm | `winget install OpenJS.NodeJS.LTS` |
| Deno (JavaScript runtime yt-dlp uses for YouTube; without it yt-dlp warns "No supported JavaScript runtime" and some formats may be missing) | `winget install DenoLand.Deno` |
| Ollama, signed in (for the cloud LLM) | install from ollama.com, then `ollama signin` |

Open a new terminal after installing so PATH is refreshed. Optional settings go in `youtube-tts-dataset/.env`
(copy `.env.example`); lookup order is environment, then `youtube-tts-dataset/.env`, then the repo-root `.env`.

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Local Ollama daemon; when signed in it routes `*-cloud` models to Ollama cloud. |
| `OLLAMA_API_KEY` | empty | Only for `OLLAMA_HOST=https://ollama.com` (direct cloud API). |
| `LLM_MODEL` | `gemma4:31b-cloud` | Punctuation and translation model. |
| `YTDS_DATA_DIR` | `youtube-tts-dataset/var` | Work data (SQLite db, downloads, stems, model cache). |
| `YTDS_EXPORT_DIR` | `<repo>/var/import` | Default export root. |
| `YTDS_PORT` | `8765` | Backend port. |
| `HF_TOKEN` | empty | Only if a Hugging Face download needs it. |

The LLM host and model can also be changed on the Settings page ("Test connection" checks them).

### LLM availability (HTTP 402)

During development a live call to `gemma4:31b-cloud` returned **HTTP 402** (Ollama cloud plan or quota).
The tool keeps working without the LLM: punctuation is skipped (clips keep the raw STT text and get the
`no_punct` flag), and clips that need translation wait in `pending_llm`. A red banner shows when the LLM is
unreachable. Once the LLM works again, press **Retry LLM** on the Review page to punctuate and translate them.

## Run

```powershell
cd youtube-tts-dataset
.\run.ps1            # checks tools, uv sync, rebuilds the frontend if its sources changed, serves :8765
.\run.ps1 -Rebuild   # force npm ci and a rebuild of frontend/dist
.\run.ps1 -Dev       # backend + Vite dev server (http://localhost:5173) in two windows
```

`run.ps1` stops with the fix to apply when ffmpeg, `nvidia-smi`, uv or npm is missing, and warns (but
continues) when Deno is missing. The backend binds to `127.0.0.1` only.

Workflow: **Jobs** — paste URLs (one per line, playlists allowed), name the dataset, Start; watch per-stage
progress. **Review** — `Space` play/pause, `Enter` approve, `R` reject, `J`/`K` next/previous; edits save
automatically. **Dataset** — kept/dropped hours, split preview, Export. **Settings** — thresholds, LLM,
cookies file for age-gated videos, read-only model list.

## Pipeline

One GPU worker holds one model at a time and works stage by stage across queued videos; downloads and CPU
stages run in a 2-thread pool alongside it. A stage is marked done only after its outputs are written, so a
restart resumes where it stopped.

| # | Stage | Tool | Device |
|---|---|---|---|
| 1 | download | yt-dlp best audio → ffmpeg 48 kHz mono WAV; title, channel, date, licence saved | network |
| 2 | separate | Demucs `htdemucs` vocals stem, 7 s segments (halved on OOM) | GPU |
| 3 | segment | Silero VAD; merge pauses < 300 ms; clips 2–15 s | CPU |
| 4 | quality | DNSMOS P.835 OVRL (drop < 3.0); ECAPA speaker similarity to the video centroid (drop < 0.60) | CPU / GPU |
| 5 | transcribe | A: `SayedShaun/Qwen3-ASR-0.6B-Bengali`; B: `mozilla-ai/whisper-large-v3-turbo-bn` (CT2 int8_float16); route by CER(A, B) | GPU |
| 6 | fallback | faster-whisper `large-v3` translate → English → LLM → Bangla; tagged `translated` | GPU + LLM |
| 7 | punctuate | LLM adds punctuation to A's text; rejected if any word changes (`no_punct`) | LLM |
| 8 | review | manual, in the UI | — |
| 9 | export | per dataset, on demand | CPU |

Routing after transcription (CER after `bn_normalize(mode="asr")`):

| Condition | Status |
|---|---|
| CER ≤ `cer_auto` | `auto` — exported as is |
| `cer_auto` < CER ≤ `cer_fallback` | `flagged` — review queue |
| CER > `cer_fallback` or A empty | fallback → `needs_review` (translated) — review queue; `pending_llm` while the LLM is down |

A video gets a "poor STT" warning when over half its clips fall back, and a "<30 s clean audio" warning
when little survives.

### Thresholds (Settings page)

| Setting | Default |
|---|---|
| `cer_auto` | 0.05 |
| `cer_fallback` | 0.30 |
| `dnsmos_min` (OVRL) | 3.0 |
| `speaker_sim_min` | 0.60 |
| `clip_min_s` / `clip_max_s` | 2.0 / 15.0 |
| `merge_gap_s` | 0.30 |
| `demucs_segment_s` | 7.0 |

## Output

Export writes to `<YTDS_EXPORT_DIR>/<dataset name>/` (default `<repo>/var/import/<dataset>/`):

```
<dataset>/
  wavs/<video_id>_<idx>.wav   24 kHz, mono, 16-bit PCM, -23 LUFS, -1 dBTP, 150 ms padding
  metadata.csv                file,text,speaker,split   (file = wavs/<name>.wav)
  curation.json               provenance: tool, git commit, thresholds, models, per-video source and kept/dropped seconds, counts, commercial_use
```

- Exported clips: `auto` plus approved clips. Flagged clips you have not approved are left out.
- Translated clips are never exported unreviewed: Export refuses with a count, or you can export without them.
- Splits are 90/5/5 train/val/test by duration, contiguous runs per video, fixed seed; a video with < 60 s
  kept goes to train only.
- Each clip is checked with the platform's `validate_audio`; failing clips are skipped and counted.
- Files are named by YouTube video id and clip index, never by title.
- The tool replaces an existing output folder only if it is a previous export (has `metadata.csv` or
  `curation.json`); any other existing folder is refused.

### Ingest into the platform

In the platform's dataset upload page choose **From server path**, pick the exported folder (it lists folders
under `var/import/` that contain a `metadata.csv`) and the `tts` task. If the platform runs on another
machine, copy the folder into that machine's `var/import/` first.

## Licence

Audio taken from YouTube is for **research and development only**. `curation.json` records each video's URL,
channel and licence field, and sets `"commercial_use": false`. Change it only after you have recorded a licence
that allows commercial use, so YouTube-derived data can be filtered out of anything commercial.

## Tests

```powershell
cd youtube-tts-dataset\backend
uv run pytest                                   # unit + API tests (gpu/network skipped)
uv run pytest -m "gpu and network" -v -s tests/test_smoke_gpu.py   # full pipeline on the GPU
cd ..\frontend
npm test
```

The smoke test runs the real pipeline on the 123 s Bangla video in `backend/tests/data/smoke_url.txt`,
approves every review clip, exports, and checks `metadata.csv`, `validate_audio` and peak VRAM.

### Smoke test results (RTX 4060 8 GB, 2026-11-26)

| Measure | Value |
|---|---|
| Video | 123 s, 24 clips: 8–10 `auto`, 13–15 `flagged`, 1 dropped (varies slightly between runs) |
| Pipeline wall time (download → punctuate) | 60.6 s (65.0 s on a second run); models already cached |
| Export wall time | 0.6 s, 23 clips |
| Peak VRAM, `torch.cuda.max_memory_allocated()` | 1789 MiB (torch allocations in the test process only) |
| Peak VRAM, `nvidia-smi` sampled every 0.5 s | 3544 MiB used, 1622 MiB baseline → **1922 MiB** for the pipeline (includes CTranslate2) |

`torch.cuda.max_memory_allocated()` does not see CTranslate2 (faster-whisper) memory, so the test also
samples `nvidia-smi` and asserts the increase over the baseline stays under 7 GB. No clip in this video
fell back to translation; loading the fallback model (`large-v3`, int8_float16) and translating 4 clips on
its own peaked at 2309 MiB over baseline. The LLM returned HTTP 402 during these runs, so every clip had
`no_punct`.
