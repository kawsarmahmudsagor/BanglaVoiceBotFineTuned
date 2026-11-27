# YTDS Studio and sentence-complete clips — design

Date: 2026-11-27. Status: approved in brainstorming, awaiting spec review.
Builds on: `2026-11-26-youtube-tts-dataset-design.md` (the existing tool, here called "the dashboard").

## 1. Goal

Add a Studio page to the YouTube TTS dataset builder. In Studio the user adds one video at a time, from a YouTube link or a local file, previews it, cuts out the parts to keep, and only then starts processing. After processing, the user reviews and edits every clip of that video on the same page and can export the dataset from there.

At the same time, change clip segmentation for every video (Studio and dashboard): clips become 15–25 s long and always end on a complete sentence.

### 1.1 Decisions from brainstorming

| Topic | Decision |
|---|---|
| Existing UI | Jobs, Review, Dataset and Settings pages stay as they are. They act as the dashboard. Nothing is removed. |
| Isolation | Work happens in the git worktree `.claude/worktrees/ytds-studio` on branch `feat/ytds-studio`. The worktree's server uses its own port and data dir, so the live server on `main` is not affected. |
| Inputs | YouTube URL (single video) or a local video file dropped on the page. |
| Start | Processing does not start on add. Download/upload first, then preview, then the user starts it. |
| Cutting | The user marks any number of keep ranges by hand on the timeline. No preset ranges. Zero ranges means the whole video. |
| Dataset | Studio has a dataset picker (or create new). Each video lands in the chosen dataset and also shows on the dashboard pages. |
| Concurrency | Studio keeps a session list of its videos. The user can queue several and come back to any. The GPU still runs one stage at a time. |
| Review edits | Text edit and approve/reject. No boundary edits, merges or splits. |
| Clip length | 15–25 s, ending on complete sentences (one or more). Applies to Studio and dashboard videos. Already-processed videos keep their clips. |
| Sentence detection | Approach A: short VAD pieces (≤10 s), transcribe, LLM punctuate, group into clips at sentence ends. |
| Transcription | Two models on every piece: model A = local Qwen3-ASR 0.6B fine-tune on Bengali.AI 200h, model B = local Whisper large-v3-turbo-bn fine-tune `r-854260e0` on Bengali.AI 200h (CTranslate2). The better transcript of each clip is kept (§4.6). Ties go to model B. |
| LLM | Punctuation only, as today. |
| Translation | Removed from the pipeline for every video. A clip whose transcript looks broken (the checks that used to send it to translation) is dropped with `drop_reason=bad_stt`. |
| Export | Unchanged. Studio's Export button calls the existing per-dataset export. |

### 1.2 Out of scope

- Editing clip boundaries, merging or splitting clips in review.
- Forced alignment or word timestamps.
- Playlists in Studio (the dashboard still expands them).
- Changing the export format.
- The English STT + LLM translation path. It is removed (§4.7).

## 2. Worktree and runtime

- Worktree: `.claude/worktrees/ytds-studio`, branch `feat/ytds-studio`, rebased onto `origin/main` at 7e5510e (includes `model-specification/STT/whisper-specs.md` and `stt-playground/whisper_chunking.py`).
- The worktree's `youtube-tts-dataset/.env` sets `YTDS_PORT=8766` and `YTDS_DATA_DIR=youtube-tts-dataset/var-studio`. The live server keeps port 8765 and `youtube-tts-dataset/var`.
- `var-studio/` is git-ignored like `var/`.
- No new Python or npm packages. `faster-whisper` and `ctranslate2` are already dependencies; pyproject pins them to `==1.2.1` and `==4.9.1`, the versions the Whisper spec's P1 fix was verified on (the lock already resolves to them). ffmpeg (already required) does the cut, the audio extract and the preview proxy.
- Model locations (new settings, env > `.env`):

| Setting | Default | Contents |
|---|---|---|
| `MODEL_A_PATH` | `C:\Users\BS23-DESKTOP-00038\Downloads\qwen-3-asr-finetune-1\safetensors` | HF safetensors checkpoint, `Qwen3ASRForConditionalGeneration` |
| `MODEL_B_PATH` | `C:\Users\BS23-DESKTOP-00038\Downloads\whisper-finetune-1\modelswhisper-v3-bn-version1` | CTranslate2 export (`model.bin`, `config.json`, `tokenizer.json`, `vocabulary.json`, `preprocessor_config.json`) |

- Each value may also be a Hugging Face repo id; it is downloaded with `HF_TOKEN` on first use. Both models are pushed as **private** repos to the user's HF account: `qwen3-asr-0.6b-bn-bengaliai200h` and `whisper-large-v3-turbo-bn-bengaliai200h-ct2`, each with a model card naming the recipe, base model, data and eval numbers. The Qwen card records `commercial_ok: false` (inherited from `SayedShaun/Qwen3-ASR-0.6B-Bengali`).
- The app checks both paths at startup and on **Start processing**. A missing model gives a clear error; no video starts.
- VRAM: model A about 1.5 GB (bf16), model B about 1.6 GB (float16). The GPU worker keeps one model loaded at a time, as today.

### 2.2 Model choice in Settings

The dashboard Settings page (not Studio) gets a **Transcription models** section with two dropdowns, **Model A** and **Model B**. Options:

| Slot | Option id | Label | Source | Default |
|---|---|---|---|---|
| A | `qwen3-asr-bn-ft1` | Qwen3-ASR 0.6B — Bengali.AI 200h fine-tune | `MODEL_A_PATH` (local folder or HF repo id) | yes |
| A | `qwen3-asr-bn-sayedshaun` | Qwen3-ASR 0.6B Bengali (SayedShaun, current model) | HF `SayedShaun/Qwen3-ASR-0.6B-Bengali` | |
| B | `whisper-bn-ft1` | Whisper large-v3-turbo-bn — Bengali.AI 200h fine-tune (`r-854260e0`) | `MODEL_B_PATH` (CT2 folder or HF repo id) | yes |
| B | `whisper-bn-mozilla` | Whisper large-v3-turbo-bn (mozilla-ai, the earlier model B) | HF `mozilla-ai/whisper-large-v3-turbo-bn`, converted to CT2 float16 once and cached under `<data_dir>/models/` (the converter removed in 5c05bfa comes back) | |

- Model A options load with the same Qwen3-ASR code; model B options with the same faster-whisper code and all §4.5 rules.
- The choice is stored in the `settings` table (`model_a`, `model_b`). `GET /api/settings` returns the options and the current choice; `PUT /api/settings` accepts `model_a` / `model_b` (unknown id: 422).
- Each option shows whether it is available (local folder present, or HF id). A local option whose folder is missing is shown disabled with the path it expected.
- Changing a model unloads it from the GPU worker if it is loaded. The new model is used from the next `transcribe_a` / `transcribe_b` run. Stages that already ran keep their text. A hint under the dropdowns says so.
- Provenance: each video records `model_a` / `model_b` option ids when its transcribe stages run, and the export's `curation.json` `models` block lists the ids (and resolved paths or repo ids) of the models used. The `fallback` entry is removed from that block.

## 3. Data model

### 3.1 `videos` table: new columns

| Column | Type | Values |
|---|---|---|
| `origin` | TEXT | `dashboard` (default) or `studio` |
| `source` | TEXT | `youtube` (default) or `local` |
| `cuts` | TEXT | JSON list of `[start_s, end_s]` keep ranges in original-video seconds; NULL means whole video |
| `media` | TEXT | relative path of the playable preview file (Studio only), NULL otherwise |
| `model_a`, `model_b` | TEXT | option ids of the models that transcribed the video (§2.2) |

Columns are added with `ALTER TABLE ... ADD COLUMN` when missing, so an existing `ytds.db` opens without loss.

### 3.2 Video statuses

New statuses: `fetching` and `awaiting_cut`. Full list: `fetching`, `awaiting_cut`, `queued`, `running`, `done`, `failed`, `skipped`.

- `fetching`: Studio download or upload in progress. Runs the `download` stage only.
- `awaiting_cut`: preview ready. The runner does nothing for the video until the user starts it.
- Dashboard videos never enter either status.

### 3.3 Clips

- New clip column `flag` (TEXT, NULL or `incomplete_sentence` or `unsplit`). It records why grouping could not end the clip on a sentence.
- New clip columns `text_pick` (`a` or `b`: which model's text was kept) and `score_a`, `score_b` (REAL). The existing `text_a`, `text_b` and `cer` columns hold the two clip transcripts and their disagreement again.
- `flag` also takes `disagree` (§4.6).
- New `pieces` table, internal to segmentation: `(dataset_id, video_id, idx, range_idx, start_s, end_s, text_a, text_b, b_tokens, b_compression, b_retry_failed, text, broken, PRIMARY KEY(dataset_id, video_id, idx))`. `text` is the kept, punctuated piece text.
- Clip `start_s`/`end_s` stay in the cut-audio timeline (§4.2), which is what the audio slicing and export read. The API adds `orig_start_s`/`orig_end_s` from the time map for display.

### 3.4 Thresholds

| Name | Old | New |
|---|---|---|
| `clip_min_s` | 2.0 | 15.0 |
| `clip_max_s` | 15.0 | 25.0 |
| `piece_max_s` | — | 10.0 |
| `piece_gap_s` | — | 0.3 (minimum pause that may split pieces) |
| `tail_min_s` | — | 5.0 |
| `disagree_cer` | — | 0.25 (A/B CER above this flags the clip) |
| `tie_margin` | — | 0.05 (score gap under this is a tie; model B wins) |

Stored thresholds that still hold the old defaults (2.0 / 15.0) are moved to the new defaults on startup. Values the user changed are kept.

## 4. Pipeline

### 4.1 Stage order

```
download → cut → separate → pieces → transcribe_a → transcribe_b → punctuate_pieces
         → group → quality → select → punctuate
```

| Stage | Thread | Does |
|---|---|---|
| `download` | pool | YouTube: dashboard downloads audio only (as today); Studio downloads video ≤720p mp4 (H.264+AAC) and extracts 48 kHz mono audio. Local: the upload already wrote the file; the stage extracts audio and makes the preview proxy if needed. |
| `cut` | pool | Writes `cut/audio.wav` and `cut/map.json` (§4.2). |
| `separate` | GPU | Demucs on `cut/audio.wav` (was `download/audio.wav`). |
| `pieces` | pool | Silero VAD on the vocals, per keep range, into pieces ≤ `piece_max_s`, split at pauses ≥ `piece_gap_s`. A piece longer than `piece_max_s` with no pause is split at the largest internal pause, or hard-split as the last resort. |
| `transcribe_a` | GPU | Model A on every piece, batched (as today's transcribe, from `MODEL_A_PATH`). Pieces are ≤10 s, which matches the short clips the fine-tune was trained on. |
| `transcribe_b` | GPU | Model B on every piece (§4.5). |
| `punctuate_pieces` | pool | Per piece, keep the better of `text_a`/`text_b` (§4.6); a piece is `broken` when both are. Then LLM punctuation over each range's kept piece texts in order (words locked, as today). Returns the punctuated text per piece. |
| `group` | pool | Builds clips (§4.3) and writes them with `replace_clips`. |
| `quality` | GPU | As today, on the new clips. |
| `select` | pool | Clip `text_a` / `text_b` = its pieces' model A / model B texts joined with spaces. Scores both, keeps the better (§4.6), sets `text_pick`, `cer`, `flag=disagree` when needed. Both broken: `status=dropped`, `drop_reason=bad_stt`. Replaces today's clip-level `transcribe`; no second GPU pass. |
| `punctuate` | pool | As today, on the kept clip text (it used to read `text_a`; now it reads the text `text_pick` names). |

The `fallback` stage is removed. The old `segment` stage is replaced by `pieces` + `transcribe_a` + `transcribe_b` + `punctuate_pieces` + `group`, and the clip-level `transcribe` stage by `select`.

Pieces are cut where the `pieces` stage cuts them: at pauses, or, for unbroken speech over 10 s, at the quietest 20 ms frame in the last 2 s before the limit (Whisper spec P3). Pieces never overlap, because clip edges are piece edges.

### 4.2 Cut

- Input: `download/audio.wav` and `cuts` (NULL means one range covering the whole file).
- Ranges are sorted, clamped to the file length, and overlapping ranges merged. Ranges under 1 s are dropped.
- Output audio: the kept ranges joined in order with 2.0 s of silence between them.
- `map.json`: list of `{range_idx, cut_start_s, cut_end_s, orig_start_s}`. A cut-timeline time `t` in range `r` maps to `orig_start_s + (t - cut_start_s)`.
- Pieces and clips carry `range_idx`; grouping never joins pieces from different ranges.

### 4.3 Grouping

Per keep range, walk the pieces in order and grow the current clip:

1. A piece "ends a sentence" when its punctuated text, stripped, ends in `।`, `?`, `!` or `.`.
2. Add the next piece while the clip stays ≤ `clip_max_s` (span from the first piece's start to the last piece's end).
3. Close the clip at the first sentence-ending piece once the clip is ≥ `clip_min_s`.
4. If adding the next piece would pass `clip_max_s`, the clip has no sentence end at ≥ `clip_min_s` (rule 3 would have closed it). Then:
   - close at the last sentence-ending piece in the clip that leaves the clip ≥ `tail_min_s` (a complete but shorter clip; sentence completeness wins over the 15 s minimum), or
   - if there is none, close at the piece boundary with the largest pause in the 15–25 s window and set `flag=incomplete_sentence`.
   The remaining pieces start the next clip.
5. A `broken` piece acts like a range edge: the clip before it closes by these same rules (as at the end of a range), and the next clip starts after it. Its audio is in no clip.
6. At the end of a range, the leftover tail becomes a clip when it is ≥ `tail_min_s` and its last piece ends a sentence. Otherwise it is stored as a clip with `status=dropped`, `drop_reason=length`.
7. A single piece longer than `clip_max_s` cannot occur (`piece_max_s` < `clip_max_s`).

Clip start/end are the first piece's start and the last piece's end, padded by 0.1 s on each side, but never into a neighbouring clip and never across a range edge.

### 4.4 LLM down

- If `punctuate_pieces` cannot reach the LLM, piece texts stay unpunctuated and `group` falls back to pause-based grouping: close each clip at the largest pause in its 15–25 s window. Each such clip gets `flag=unsplit` and status `needs_review` once routed.
- **Retry LLM** for a dataset reruns `punctuate_pieces` and `group` for videos with `unsplit` clips, but only when no clip of that video has been edited or approved/rejected. Otherwise those clips stay as they are.

### 4.5 Model B decoding (from `model-specification/STT/whisper-specs.md`)

- Load with faster-whisper `WhisperModel(MODEL_B_PATH, device="cuda", compute_type="float16")`, then set `model.max_length = 880` (P1: lifts the 224-token cap to 440; never higher).
- Decode all pieces of a video through `BatchedInferencePipeline.transcribe(audio, clip_timestamps=[...], vad_filter=False, language="bn", without_timestamps=True, batch_size=16, beam_size=5)`, one window per piece. Pieces already follow P2 (≤10 s, split at pauses) and P3 (quietest-frame split).
- P4 retry: a piece window that produced ≥ `cap - 2` tokens, or is ≥2.5 s long with fewer than 12 tokens per second, is decoded again as ≤5 s sub-windows (VAD regions inside it, quietest-frame splits, 1 s overlap, `join_overlap` dedupe). The retry is kept only if it has more tokens and a compression ratio ≤ 2.4.
- The windowing helpers are ported from `stt-playground/whisper_chunking.py` (`plan_windows`, `_quietest_split`, `join_overlap`, `_suspicious`, `_decode`) into `ytds/pipeline/whisper_b.py`, without the playground's logging and monkey-patching.
- Per piece, store `text_b`, its token count, compression ratio and whether a P4 retry failed (used by §4.6). No period is appended to results (P5 does not apply: no RealtimeSTT here).
- The Whisper spec's "required checks after an upgrade" become a marked GPU test: an 18 s Bangla clip stops at 224 tokens with `max_length=448` and ends naturally under 440 with 880.

### 4.6 Model selection

Each transcript (piece or clip) gets a score in [0, 1]; 0 means broken.

- Broken (`route()` says `broken`): 0.
- Otherwise start at 1.0 and subtract:
  - `1 - bn_ratio` (share of non-Bangla letters);
  - 0.1 per repeat above 2 in `longest_repeat`;
  - for a speaking rate outside 8–20 Bangla characters per second: 0.02 per char/s outside that band, at most 0.3;
  - model B only: 0.3 when any window's compression ratio > 2.4, and 0.2 when a window was flagged by P4 and its retry was not kept.
- Clamp to [0, 1]. Keep the higher score. If the gap is under `tie_margin`, keep model B.
- At clip level, compute the CER between the two texts (`bn_normalize(..., mode="asr")`, punctuation removed). Above `disagree_cer`, set `flag=disagree` and status `flagged`, so the clip goes to review with both texts shown.
- Review: the card shows model A and model B text side by side with the existing diff view (today used for `reason="cer"`), marks the kept one, and has **Use model A** / **Use model B**. That writes the chosen model's text into `text_final` and sets `text_pick`; approval stays with the reviewer.
- Clips without `disagree` keep status `auto` and are exported as today.

### 4.7 Broken transcripts (translation removed)

- `route()` keeps its checks (empty text, Bangla-script share below `min_bn_ratio`, speaking rate outside `min/max_chars_per_s`, a word repeated `max_word_repeat` times). Its `fallback` result is renamed `broken`.
- A piece is broken when both its model A and model B texts are broken. It is excluded from grouping (§4.3 rule 5).
- At `select`, a clip whose two texts are both broken gets `status=dropped`, `drop_reason=bad_stt`. It is shown under the Dropped filter and can still be played.
- The `fallback` stage, its `whisper_translate` GPU model and the `pending_llm` status are no longer produced. The fallback module and loader are deleted.
- The video warning "poor STT" now means more than half of the clips were dropped as `bad_stt`.
- **Retry LLM** only reruns punctuation (and §4.4 regrouping). It no longer translates.
- Settings hints for the three routing thresholds say "the clip is dropped" instead of "translated". Nothing else on the Settings page is removed; the model dropdowns (§2.2) are added.
- Old datasets: existing `translated` and `pending_llm` clips are left as they are. The dashboard Review page still shows and edits them, and export still treats them as today. Nothing new of that kind is created.

### 4.8 Migration of existing videos

On startup, every video whose `segment` stage is done gets `cut`, `pieces`, `transcribe_a`, `transcribe_b`, `punctuate_pieces` and `group` marked done, and every video whose `transcribe` stage is done gets `select` marked done. Their existing clips stay. The old `segment` marks are left in place and ignored.

## 5. Runner changes

- `fetching` counts as active for the `download` stage only. When `download` finishes on a Studio video, the runner sets `awaiting_cut` instead of moving on.
- `awaiting_cut` is not active. `POST .../start` writes `cuts`, sets `queued` and enqueues.
- Dashboard videos run straight through, with `cut` using the whole file.
- New GPU stages `transcribe_a`, `transcribe_b`; new pool stages `cut`, `pieces`, `punctuate_pieces`, `group`, `select`. Removed: `segment`, `transcribe`, `fallback`. Stage-major scheduling means each model loads once per batch of ready videos. Stage-major GPU scheduling is unchanged.
- Stop/Delete behave as today in every status. Retry on a failed Studio video at `download` returns it to `fetching`.

## 6. API

All under `/api/datasets/{dataset_id}`.

| Method | Path | Does |
|---|---|---|
| POST | `/studio/videos` | Body `{url}`. Expands one YouTube video (playlists rejected with 400), adds it with `origin=studio`, status `fetching`. |
| POST | `/studio/upload` | Multipart file, streamed to disk in 1 MB chunks. Video id `local_<first 11 hex of sha1>`; same file again is a duplicate (409 with the existing id). Title is the file name. Status `fetching`. |
| GET | `/studio/videos` | Studio videos of the dataset with status, stages and clip counts. |
| GET | `/videos/{video_id}/media` | The preview file, with HTTP Range support (FileResponse). 404 until ready. |
| POST | `/videos/{video_id}/start` | Body `{cuts: [[start_s, end_s], ...]}`. Only in `awaiting_cut` (409 otherwise). Validates ranges (start < end, within duration). Empty list means whole video. |
| GET | `/videos/{video_id}/clips` | All clips of the video (every status, dropped included), each with `reason`, `flag`, `orig_start_s`, `orig_end_s`. |

Clip text edits and approve/reject use the existing `PATCH /clips/{clip_id}`. Clip audio uses the existing `/clips/{clip_id}/audio`. Export uses the existing `POST /export`.

Preview proxy: if a local file is not `.mp4`/`.webm`, or ffprobe reports a video codec other than h264/vp8/vp9/av1, ffmpeg writes `media/preview.mp4` (720p H.264, AAC). Otherwise the original is served.

## 7. Studio UI

- Route: real path `/studio` (the backend serves `index.html` for it) and hash `#/studio`. A "Studio" nav link is added; other pages are unchanged.
- Layout: left column, main panel.
- Left column: dataset picker with "New dataset", a URL box, a drop zone (also click-to-browse), and the session list of Studio videos in the chosen dataset with status, current stage and progress from the existing WebSocket.
- Main panel, by the selected video's status:
  - `fetching`: download or upload progress (upload progress from the browser's XHR upload events).
  - `awaiting_cut`: `<video>` player with full controls and a range timeline under it. "Mark in" and "Mark out" buttons add a range at the playhead. Ranges can be dragged at both edges, selected, played, and deleted. Start/end times are also typed fields. Total kept duration is shown. **Start processing** calls `/start`.
  - `queued`/`running`: stage list with per-stage progress, Stop button.
  - `done`: clip review list built from the existing `ReviewCard` (play clip audio, edit text, approve/reject). Filters: All, Needs attention (flagged, needs_review, pending_llm, any `flag`), Approved, Rejected, Dropped. Each card shows original-video times and the `flag` badge. An **Export dataset** button triggers the existing export and shows its result.
  - `failed`: the error and a Retry button.
- Clips become reviewable as soon as `punctuate` finishes for the video; Studio does not wait for other videos.

## 8. Errors

| Case | Behaviour |
|---|---|
| YouTube download fails | Video `failed` with the last yt-dlp line; Retry. |
| Upload interrupted | Partial file deleted; no video row is kept. |
| File has no audio stream | Video `failed`: "no audio track". |
| Preview proxy fails | Video still goes to `awaiting_cut` without `media`; the UI shows "preview unavailable" and the user can type ranges. |
| Invalid cuts | 422 with the reason; status unchanged. |
| LLM down | §4.4. |
| No speech in kept ranges | Video `done` with the existing "<30 s clean audio" warning. |

## 9. Testing

Backend (pytest, no GPU unless marked):

- `group`: sentence end closes at ≥15 s; at the 25 s cap with only an earlier sentence end, closes there (clip 5–15 s, no flag); no sentence end gives `incomplete_sentence` at the largest pause; tails kept or dropped by §4.3 rule 6; no clip crosses a range; padding never overlaps.
- `pieces`: split at pauses, max length, hard split.
- `cut`: range normalisation, silence gaps, time map round trip.
- Store: column migration on an old DB, stage-mark migration, threshold migration keeps user values.
- Selection: score rules of §4.6 (each penalty, clamp, tie to model B); piece broken only when both texts are; clip `disagree` above `disagree_cer`; "Use model A/B" patch sets `text_final` and `text_pick`.
- Model B windowing: `plan_windows`, `join_overlap`, `_suspicious` unit tests ported with the code; marked GPU test for the P1 token cap (§4.5).
- Routing: broken piece excluded from grouping and splits clips; broken clip dropped with `bad_stt`; no clip reaches `pending_llm`; `fallback` stage not in `STAGES`.
- Runner: Studio video stops at `awaiting_cut`; `/start` resumes it; dashboard video runs straight through; LLM-down path flags `unsplit`.
- Settings: model options listed with availability; `PUT` saves a valid id and rejects an unknown one; a change unloads the old model; the next transcribe stage uses the new one; video records the ids.
- API: upload id dedupe, playlist rejected, `/start` validation and 409, media Range response, clips listing with original times.
- GPU smoke test (marked): one short video through the full new pipeline; every kept clip is 5–25 s, most are 15–25 s, and its final text ends in a sentence terminator, or the clip carries a `flag`.

Frontend (Vitest):

- Settings model dropdowns: show options, disable a missing local model, save the choice.

- Range editor: mark in/out, drag edges, delete, overlapping ranges merged on submit.
- Studio main panel renders the right view for each status.
- Review filter "Needs attention" includes flagged clips.
