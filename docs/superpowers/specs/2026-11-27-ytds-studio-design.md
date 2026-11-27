# YTDS Studio and sentence-complete clips — design

Date: 2026-11-27. Status: approved in brainstorming, awaiting spec review.
Builds on: `2026-11-26-youtube-tts-dataset-design.md` (the existing tool, here called "the dashboard").

## 1. Goal

Add a Studio page to the YouTube TTS dataset builder. In Studio the user adds one video at a time, from a YouTube link or a local file, previews it, cuts out the parts to keep, and only then starts processing. After processing, the user reviews and edits every clip of that video on the same page and can export the dataset from there.

At the same time, change clip segmentation for every video (Studio and dashboard): clips become 15–30 s long and always end on a complete sentence.

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
| Clip length | 15–30 s, ending on complete sentences (one or more). Applies to Studio and dashboard videos. Already-processed videos keep their clips. |
| Sentence detection | Approach A: short VAD pieces, transcribe, LLM punctuate, group into clips at sentence ends, transcribe each clip again as a whole. |
| Export | Unchanged. Studio's Export button calls the existing per-dataset export. |

### 1.2 Out of scope

- Editing clip boundaries, merging or splitting clips in review.
- Forced alignment or word timestamps.
- Playlists in Studio (the dashboard still expands them).
- Changing the export format.

## 2. Worktree and runtime

- Worktree: `.claude/worktrees/ytds-studio`, branch `feat/ytds-studio`, created from `main` at 3ec2ce9.
- The worktree's `youtube-tts-dataset/.env` sets `YTDS_PORT=8766` and `YTDS_DATA_DIR=youtube-tts-dataset/var-studio`. The live server keeps port 8765 and `youtube-tts-dataset/var`.
- `var-studio/` is git-ignored like `var/`.
- No new Python or npm dependencies are planned. ffmpeg (already required) does the cut, the audio extract and the preview proxy.

## 3. Data model

### 3.1 `videos` table: new columns

| Column | Type | Values |
|---|---|---|
| `origin` | TEXT | `dashboard` (default) or `studio` |
| `source` | TEXT | `youtube` (default) or `local` |
| `cuts` | TEXT | JSON list of `[start_s, end_s]` keep ranges in original-video seconds; NULL means whole video |
| `media` | TEXT | relative path of the playable preview file (Studio only), NULL otherwise |

Columns are added with `ALTER TABLE ... ADD COLUMN` when missing, so an existing `ytds.db` opens without loss.

### 3.2 Video statuses

New statuses: `fetching` and `awaiting_cut`. Full list: `fetching`, `awaiting_cut`, `queued`, `running`, `done`, `failed`, `skipped`.

- `fetching`: Studio download or upload in progress. Runs the `download` stage only.
- `awaiting_cut`: preview ready. The runner does nothing for the video until the user starts it.
- Dashboard videos never enter either status.

### 3.3 Clips

- New clip column `flag` (TEXT, NULL or `incomplete_sentence` or `unsplit`). It records why grouping could not end the clip on a sentence.
- New `pieces` table, internal to segmentation: `(dataset_id, video_id, idx, range_idx, start_s, end_s, text, PRIMARY KEY(dataset_id, video_id, idx))`. `text` is the punctuated piece text.
- Clip `start_s`/`end_s` stay in the cut-audio timeline (§4.2), which is what the audio slicing and export read. The API adds `orig_start_s`/`orig_end_s` from the time map for display.

### 3.4 Thresholds

| Name | Old | New |
|---|---|---|
| `clip_min_s` | 2.0 | 15.0 |
| `clip_max_s` | 15.0 | 30.0 |
| `piece_max_s` | — | 10.0 |
| `piece_gap_s` | — | 0.3 (minimum pause that may split pieces) |
| `tail_min_s` | — | 5.0 |

Stored thresholds that still hold the old defaults (2.0 / 15.0) are moved to the new defaults on startup. Values the user changed are kept.

## 4. Pipeline

### 4.1 Stage order

```
download → cut → separate → pieces → transcribe_pieces → punctuate_pieces
         → group → quality → transcribe → fallback → punctuate
```

| Stage | Thread | Does |
|---|---|---|
| `download` | pool | YouTube: dashboard downloads audio only (as today); Studio downloads video ≤720p mp4 (H.264+AAC) and extracts 48 kHz mono audio. Local: the upload already wrote the file; the stage extracts audio and makes the preview proxy if needed. |
| `cut` | pool | Writes `cut/audio.wav` and `cut/map.json` (§4.2). |
| `separate` | GPU | Demucs on `cut/audio.wav` (was `download/audio.wav`). |
| `pieces` | pool | Silero VAD on the vocals, per keep range, into pieces ≤ `piece_max_s`, split at pauses ≥ `piece_gap_s`. A piece longer than `piece_max_s` with no pause is split at the largest internal pause, or hard-split as the last resort. |
| `transcribe_pieces` | GPU | Model A on every piece. |
| `punctuate_pieces` | pool | LLM punctuation over each range's piece texts in order (words locked, as today). Returns the punctuated text per piece. |
| `group` | pool | Builds clips (§4.3) and writes them with `replace_clips`. |
| `quality`, `transcribe`, `fallback`, `punctuate` | as today | Run on the new 15–30 s clips. `transcribe` uses the whole clip audio. Model A already caps at 30 s. |

The old `segment` stage is replaced by `pieces` + `transcribe_pieces` + `punctuate_pieces` + `group`.

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
   - if there is none, close at the piece boundary with the largest pause in the 15–30 s window and set `flag=incomplete_sentence`.
   The remaining pieces start the next clip.
5. At the end of a range, the leftover tail becomes a clip when it is ≥ `tail_min_s` and its last piece ends a sentence. Otherwise it is stored as a clip with `status=dropped`, `drop_reason=length`.
6. A single piece longer than `clip_max_s` cannot occur (`piece_max_s` < `clip_max_s`).

Clip start/end are the first piece's start and the last piece's end, padded by 0.1 s on each side, but never into a neighbouring clip and never across a range edge.

### 4.4 LLM down

- If `punctuate_pieces` cannot reach the LLM, piece texts stay unpunctuated and `group` falls back to pause-based grouping: close each clip at the largest pause in its 15–30 s window. Each such clip gets `flag=unsplit` and status `needs_review` once routed.
- **Retry LLM** for a dataset reruns `punctuate_pieces` and `group` for videos with `unsplit` clips, but only when no clip of that video has been edited or approved/rejected. Otherwise those clips stay as they are.

### 4.5 Migration of existing videos

On startup, every video whose `segment` stage is done gets `cut`, `pieces`, `transcribe_pieces`, `punctuate_pieces` and `group` marked done. Their existing clips stay. The old `segment` marks are left in place and ignored.

## 5. Runner changes

- `fetching` counts as active for the `download` stage only. When `download` finishes on a Studio video, the runner sets `awaiting_cut` instead of moving on.
- `awaiting_cut` is not active. `POST .../start` writes `cuts`, sets `queued` and enqueues.
- Dashboard videos run straight through, with `cut` using the whole file.
- New GPU stages `transcribe_pieces`; new pool stages `cut`, `pieces`, `punctuate_pieces`, `group`. Stage-major GPU scheduling is unchanged.
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

- `group`: sentence end closes at ≥15 s; at the 30 s cap with only an earlier sentence end, closes there (clip 5–15 s, no flag); no sentence end gives `incomplete_sentence` at the largest pause; tails kept or dropped by §4.3 rule 5; no clip crosses a range; padding never overlaps.
- `pieces`: split at pauses, max length, hard split.
- `cut`: range normalisation, silence gaps, time map round trip.
- Store: column migration on an old DB, stage-mark migration, threshold migration keeps user values.
- Runner: Studio video stops at `awaiting_cut`; `/start` resumes it; dashboard video runs straight through; LLM-down path flags `unsplit`.
- API: upload id dedupe, playlist rejected, `/start` validation and 409, media Range response, clips listing with original times.
- GPU smoke test (marked): one short video through the full new pipeline; every kept clip is 5–30 s, most are 15–30 s, and its final text ends in a sentence terminator, or the clip carries a `flag`.

Frontend (Vitest):

- Range editor: mark in/out, drag edges, delete, overlapping ranges merged on submit.
- Studio main panel renders the right view for each status.
- Review filter "Needs attention" includes flagged clips.
