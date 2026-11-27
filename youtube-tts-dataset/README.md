# YouTube TTS dataset builder (`ytds`)

Turns YouTube videos or local video files of **one Bangla speaker** into a TTS voice dataset in the platform's
`tts` layout (`wavs/` + `metadata.csv` + `curation.json`). It downloads the audio, removes music, cuts
15–25 s clips that end on complete sentences, drops noisy clips and other voices, transcribes with two Bangla
STT models and keeps the better text, lets you review doubtful clips in a web UI, and exports. Progress for
every stage is shown live.

Two ways in: the **dashboard** pages (Jobs, Review, Dataset, Settings: paste URLs, playlists allowed, every
video runs straight through) and the **Studio** page (one video at a time, YouTube or a local file: preview it,
mark the parts to keep, then start; review and export on the same page).

Runs on a Windows 11 PC with an NVIDIA GPU (built and tested on an RTX 4060 8 GB), Windows PowerShell 5.1.
Design: [`2026-11-26-youtube-tts-dataset-design.md`](../docs/superpowers/specs/2026-11-26-youtube-tts-dataset-design.md)
and, for Studio and sentence-complete clips,
[`2026-11-27-ytds-studio-design.md`](../docs/superpowers/specs/2026-11-27-ytds-studio-design.md).

## Setup

| Need | Install |
|---|---|
| ffmpeg on PATH | `winget install Gyan.FFmpeg` |
| NVIDIA driver (`nvidia-smi` works) | from nvidia.com |
| uv | `winget install astral-sh.uv` |
| Node.js 20+ and npm | `winget install OpenJS.NodeJS.LTS` |
| Deno (JavaScript runtime yt-dlp uses for YouTube; without it yt-dlp warns "No supported JavaScript runtime" and some formats may be missing) | `winget install DenoLand.Deno` |
| Ollama with the punctuation model | install from ollama.com, then `ollama pull gemma4:e4b` (`ollama signin` only for `*-cloud` models) |
| The two transcription models | local folders, see `MODEL_A_PATH` / `MODEL_B_PATH` below |

Open a new terminal after installing so PATH is refreshed. Optional settings go in `youtube-tts-dataset/.env`
(copy `.env.example`); lookup order is environment, then `youtube-tts-dataset/.env`, then the repo-root `.env`.

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Local Ollama daemon; when signed in it routes `*-cloud` models to Ollama cloud. |
| `OLLAMA_API_KEY` | empty | Only for `OLLAMA_HOST=https://ollama.com` (direct cloud API). |
| `LLM_MODEL` | `gemma4:e4b` | Punctuation model (local Ollama). |
| `YTDS_DATA_DIR` | `youtube-tts-dataset/var` | Work data (SQLite db, downloads, stems, model cache). |
| `YTDS_EXPORT_DIR` | `<repo>/var/import` | Default export root. |
| `YTDS_PORT` | `8765` | Backend port. |
| `HF_TOKEN` | empty | Only if a Hugging Face download needs it. |
| `MODEL_A_PATH` | `C:\Users\BS23-DESKTOP-00038\Downloads\qwen-3-asr-finetune-1\safetensors` | Model A: Qwen3-ASR 0.6B Bengali.AI 200h fine-tune (HF safetensors folder, or an HF repo id). |
| `MODEL_B_PATH` | `C:\Users\BS23-DESKTOP-00038\Downloads\whisper-finetune-1\modelswhisper-v3-bn-version1` | Model B: Whisper large-v3-turbo-bn fine-tune `r-854260e0`, CTranslate2 export (`model.bin`, `config.json`, `tokenizer.json`, `vocabulary.json`, `preprocessor_config.json`), or an HF repo id. |

The LLM host and model can also be changed on the Settings page ("Test connection" checks them).

### Transcription models (Settings page)

The Settings page has a **Transcription models** section with two dropdowns:

| Dropdown | Option | Source |
|---|---|---|
| Model A | Qwen3-ASR 0.6B — Bengali.AI 200h fine-tune (default) | `MODEL_A_PATH` |
| Model A | Qwen3-ASR 0.6B Bengali (SayedShaun, the earlier model) | HF `SayedShaun/Qwen3-ASR-0.6B-Bengali` |
| Model B | Whisper large-v3-turbo-bn — Bengali.AI 200h fine-tune (default) | `MODEL_B_PATH` |
| Model B | Whisper large-v3-turbo-bn (mozilla-ai, the earlier model B) | HF `mozilla-ai/whisper-large-v3-turbo-bn`, converted to CT2 float16 once into `<data_dir>/models/` |

A local option whose folder is missing is shown disabled with the path it expected, and **Start processing**
refuses with that path. A change takes effect from the next `transcribe_a` / `transcribe_b` run; a stage
already running finishes with the model it started with. Each video records the option ids that transcribed
it, and `curation.json` lists them.

### When the LLM is down

The LLM only adds punctuation (no translation anywhere). Without it the tool keeps working: pieces stay
unpunctuated, so clips are cut at the largest pause in their 15–25 s window and get `flag=unsplit` (they go
to review), and clip texts keep the raw STT text with the `no_punct` flag. A red banner shows when the LLM is
unreachable. Once it works again, press **Retry LLM** on the Review page: videos with `unsplit` clips that
nobody has reviewed yet are regrouped, the rest are only punctuated.

## Run

```powershell
cd youtube-tts-dataset
.\run.ps1            # checks tools, uv sync, rebuilds the frontend if its sources changed, serves :8765
.\run.ps1 -Rebuild   # force npm ci and a rebuild of frontend/dist
.\run.ps1 -Dev       # backend + Vite dev server (http://localhost:5173) in two windows
```

`run.ps1` stops with the fix to apply when ffmpeg, `nvidia-smi`, uv or npm is missing, and warns (but
continues) when Deno is missing. The backend binds to `127.0.0.1` only. `run.ps1` serves on `YTDS_PORT`
from the environment or `youtube-tts-dataset/.env`.

**Worktree note.** A git worktree of this repo (for example `.claude/worktrees/ytds-studio`) should set
`YTDS_PORT=8766` and `YTDS_DATA_DIR=youtube-tts-dataset/var-studio` in its own `youtube-tts-dataset/.env`, so
its server (http://127.0.0.1:8766) runs next to the main checkout's server on 8765 without sharing a database.
`var-studio/` is git-ignored like `var/`.

Workflow (dashboard): **Jobs** — paste URLs (one per line, playlists allowed), name the dataset, Start; watch
per-stage progress. **Review** — `Space` play/pause, `Enter` approve, `R` reject, `J`/`K` next/previous; edits
save automatically; a `disagree` clip shows both transcripts with **Use model A** / **Use model B**.
**Dataset** — kept/dropped hours, split preview, Export. **Settings** — thresholds, LLM, transcription models,
cookies file for age-gated videos, read-only model list.

### Studio

Open http://127.0.0.1:8765/studio (or the **Studio** link, `#/studio`).

1. Pick a dataset or create one. Studio videos also show on the dashboard pages.
2. Add one video: paste a YouTube link (single video; playlists are refused), or drop a local video file on
   the drop zone (or click it to browse). The same file dropped twice is refused as a duplicate. YouTube
   videos download at ≤720p for the preview; local files that a browser cannot play (for example `.mkv`, or
   a codec other than h264/vp8/vp9/av1) get a 720p H.264 preview proxy.
3. When the preview is ready (`awaiting_cut`), play it and mark keep ranges: **Mark in** / **Mark out** at the
   playhead, drag range edges, or type start/end times. No ranges means the whole video. Ranges are sorted,
   clamped and merged on the server; ranges under 1 s are dropped.
4. **Start processing**. The kept ranges are joined with 2 s of silence and the pipeline runs; a clip never
   spans two ranges. You can queue several videos and come back to any of them.
5. Review the clips of the video on the same page: play, edit text, approve/reject, **Use model A/B** on a
   `disagree` clip. Filters: All, Needs attention, Approved, Rejected, Dropped. Times shown are original-video
   times.
6. **Export dataset** runs the usual per-dataset export.

## Pipeline

One GPU worker holds one model at a time and works stage by stage across ready videos (each model loads once
per batch); downloads and CPU stages run in a 2-thread pool alongside it. The punctuation LLM (local Ollama)
also needs VRAM, so its stages run on the GPU thread: the torch/CT2 model is released before an LLM batch,
and Ollama is told to unload its model (`keep_alive: 0`) after it. A stage is marked done only after its
outputs are written, so a restart resumes where it stopped.

| # | Stage | Does | Device |
|---|---|---|---|
| 1 | download | Dashboard: yt-dlp best audio. Studio: also the ≤720p video for the preview; a local upload is probed, its audio extracted and a preview proxy made if needed. 48 kHz mono WAV. | network / CPU |
| 2 | cut | Keeps the Studio ranges (whole file for dashboard videos), joined with 2 s silence; writes the time map | CPU |
| 3 | separate | Demucs `htdemucs` vocals stem, 7 s segments (halved on OOM) | GPU |
| 4 | pieces | Silero VAD into speech pieces ≤ `piece_max_s` (10 s), split at pauses ≥ `piece_gap_s`, or at the quietest 20 ms frame | CPU |
| 5 | transcribe_a | Model A (Qwen3-ASR, bf16) on every piece, batched | GPU |
| 6 | transcribe_b | Model B (Whisper turbo-bn, CT2 float16) on every piece: `max_length=880`, batched decode, ≤5 s retry of windows that look cut | GPU |
| 7 | punctuate_pieces | Per piece keep the better of the A/B texts; LLM punctuation, rejected if any word changes | GPU thread + LLM |
| 8 | group | Builds 15–25 s clips that end on a sentence (rules below) | CPU |
| 9 | quality | DNSMOS P.835 OVRL (drop < 3.0); ECAPA speaker similarity to the video centroid (drop < 0.60) | CPU / GPU |
| 10 | select | Clip texts from its pieces' A and B texts; score both, keep the better (ties: B); CER > `disagree_cer` → `flag=disagree`, to review; both broken → dropped `bad_stt` | CPU |
| 11 | punctuate | LLM punctuation of the kept clip text, words locked (`no_punct` when rejected) | GPU thread + LLM |
| — | review | manual, in the UI | — |
| — | export | per dataset, on demand | CPU |

### Clip length rules

Per keep range, pieces are added to the current clip in order:

- A clip closes at the first piece that ends a sentence (`।`, `?`, `!`, `.`) once it is ≥ `clip_min_s` (15 s).
- It never grows past `clip_max_s` (25 s). If no sentence end came by then, it closes at the last sentence end
  that leaves it ≥ `tail_min_s` (5 s); if there is none, at the largest pause in its 15–25 s window with
  `flag=incomplete_sentence` (or `unsplit` when the LLM was down).
- A broken piece (both transcripts fail the checks below) is in no clip; the clips on each side of it close as
  at a range edge.
- The leftover tail of a range is a clip when it is ≥ 5 s and ends a sentence; otherwise it is dropped
  (`drop_reason=length`).
- Clip edges are padded by 0.1 s, never into a neighbouring clip or across a range edge.

So every kept clip is 5–25 s (plus padding), most are 15–25 s, and each ends on a sentence or carries a flag.

### Transcript checks

A transcript is **broken** when any check fails; its score is 0:

| Check | Fails when |
|---|---|
| empty | no text after `bn_normalize(mode="asr")` |
| script | Bangla characters are less than `min_bn_ratio` of the letters (English loanwords in Latin script pass) |
| speaking rate | characters per second outside `min_chars_per_s`–`max_chars_per_s` (missed words or made-up text) |
| loop | one word repeated `max_word_repeat` times in a row |

Otherwise the score starts at 1.0 and loses the non-Bangla share, 0.1 per word repeat above 2, up to 0.3 for a
speaking rate outside 8–20 chars/s, and (model B) 0.3 for a compression ratio > 2.4 or 0.2 for a failed retry.

**Translation is removed.** The English STT + LLM translation fallback and its `whisper_translate` model are
gone: a clip whose two transcripts are both broken is dropped with `drop_reason=bad_stt` (still playable
under the Dropped filter). Old datasets keep their `translated` and `pending_llm` clips; Review and Export
treat them as before, and nothing new of that kind is made.

A video gets a "poor STT" warning when over half its clips are dropped as `bad_stt`, and a "<30 s clean audio"
warning when little survives.

### Thresholds (Settings page)

| Setting | Default |
|---|---|
| `min_bn_ratio` | 0.5 |
| `min_chars_per_s` / `max_chars_per_s` | 3.0 / 30.0 |
| `max_word_repeat` | 6 |
| `dnsmos_min` (OVRL) | 3.0 |
| `speaker_sim_min` | 0.60 |
| `clip_min_s` / `clip_max_s` | 15.0 / 25.0 |
| `piece_max_s` | 10.0 |
| `piece_gap_s` | 0.3 |
| `tail_min_s` | 5.0 |
| `disagree_cer` | 0.25 |
| `tie_margin` | 0.05 |
| `demucs_segment_s` | 7.0 |

Stored thresholds that still held the old clip defaults (2.0 / 15.0) move to 15.0 / 25.0 on startup; values
you changed are kept. Videos processed before this change keep their clips.

## Output

Export writes to `<YTDS_EXPORT_DIR>/<dataset name>/` (default `<repo>/var/import/<dataset>/`):

```
<dataset>/
  wavs/<video_id>_<idx>.wav   24 kHz, mono, 16-bit PCM, -23 LUFS, -1 dBTP, 150 ms padding
  metadata.csv                file,text,speaker,split   (file = wavs/<name>.wav)
  curation.json               provenance: tool, git commit, thresholds, models, per-video source and kept/dropped seconds, counts, commercial_use
```

- Exported clips: `auto` plus approved clips. Flagged clips you have not approved are left out.
- Translated clips (old datasets only) are never exported unreviewed: Export refuses with a count, or you can
  export without them.
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
uv run pytest                                          # unit + API tests (gpu/network skipped)
$env:PYTHONUTF8 = "1"                                  # Bangla in the -s output
uv run pytest -m gpu -q -s tests/test_whisper_gpu.py   # model B token cap (P1), ~10 s
uv run pytest -m gpu -q -s tests/test_smoke_gpu.py     # whole pipeline on the GPU, ~5.5 min
cd ..\frontend
npm test
npm run typecheck
```

Run the GPU tests one at a time. Both need network (YouTube) and the local Ollama with `gemma4:e4b`.

- `test_whisper_gpu.py` decodes an 18 s window of the smoke video (100–118 s) with model B: 224 tokens (the
  cap) with `max_length=448`, more than 224 and under 440 with 880. (Copies of `bn_clip.wav` cannot be used:
  the fine-tune ends the decode after one sentence, 121 tokens however many copies.)
- `test_smoke_gpu.py` runs the real runner on two videos at once: the 123 s video in
  `backend/tests/data/smoke_url.txt` as a dashboard video, and a ~74 s mp4 made of `bn_clip.wav` copies as a
  Studio upload, started with one range (2–64 s). It checks the clip rules (every kept clip 5–25 s, at least
  half 15–25 s, each ends on a sentence or carries a flag or `no_punct`), exports the dashboard dataset
  (`metadata.csv`, `validate_audio`) and peak VRAM. It also prints, without asserting, the comparison check
  (model A on each whole clip vs its joined piece texts) and how fast Ollama frees its VRAM after the unload.

### GPU test results (RTX 4060 8 GB, 2026-11-27)

| Measure | Value |
|---|---|
| Model B token cap, 18 s window | 224 tokens at `max_length=448`, 264 at 880; load + 2 decodes 4.0 s; ~2.2 GB VRAM |
| Smoke clips, dashboard video | 7 clips, all kept, 15.2–17.8 s, all `auto`, all end on `।`/`?` |
| Smoke clips, Studio video | 4 clips: 3 kept (16.0, 16.3, 12.2 s, `auto`), 1 tail dropped (`length`) |
| Pipeline wall time (both videos, download → punctuate) | 272.8 s, models cached (the LLM, `gemma4:e4b`, runs partly on the CPU) |
| Export | 0.4 s, 7 clips, 0 validation drops |
| Peak VRAM, `torch.cuda.max_memory_allocated()` | 1789 MiB (torch models in the test process) |
| Peak VRAM, `nvidia-smi` sampled every 0.5 s | 7488 MiB used, 2640 MiB baseline → **4848 MiB** (the Ollama model; CT2 model B ~2.2 GB) |
| Ollama VRAM after `keep_alive: 0` | freed before the call returns (call 0.00–0.02 s; first sample after it already at baseline) |
| Comparison check (model A whole clip vs joined pieces) | mean CER 0.62 over 10 clips; whole-clip text is never longer, it is usually much shorter: model A stops after a sentence or two on a 15–25 s clip, so piece-level transcription is needed |

