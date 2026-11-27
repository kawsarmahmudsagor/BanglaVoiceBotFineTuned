# YTDS Studio and Sentence-Complete Clips Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `/studio` page (YouTube or local video → preview → user keep-ranges → process → per-clip review → export) and change the pipeline for every video to 15–25 s sentence-complete clips transcribed by two models, keeping the better text, with no translation path.

**Architecture:** New pipeline stages replace `segment`/`transcribe`/`fallback`: `cut` (keep ranges to one audio file plus a time map), `pieces` (VAD pieces ≤10 s), `transcribe_a`/`transcribe_b` (Qwen fine-tune and Whisper fine-tune on pieces), `punctuate_pieces` (LLM, kept text), `group` (sentence-end grouping), `select` (clip-level A/B pick). Studio videos pause at a new `awaiting_cut` status until the user starts them. A new Studio API module and React page sit beside the unchanged dashboard pages; Settings gains model A/B dropdowns.

**Tech Stack:** Python 3.12, FastAPI, SQLite, transformers 5.24.1 (Qwen3-ASR), faster-whisper 1.2.1 / ctranslate2 4.9.1, silero-vad, ffmpeg, yt-dlp; React 18 + TypeScript + Vite + Vitest.

**Spec:** `docs/superpowers/specs/2026-11-27-ytds-studio-design.md` (read it with this plan; § numbers below refer to it).

All paths below are relative to `youtube-tts-dataset/` unless they start with `docs/`. Run backend commands from `youtube-tts-dataset/backend` with `uv run`, frontend commands from `youtube-tts-dataset/frontend`. Shell is Windows PowerShell 5.1 or Git Bash.

## Global Constraints

- Work only in the worktree `.claude/worktrees/ytds-studio`, branch `feat/ytds-studio`. Never touch the main checkout or its `youtube-tts-dataset/var/`.
- Worktree server: `YTDS_PORT=8766`, `YTDS_DATA_DIR=youtube-tts-dataset/var-studio`. `var-studio/` is git-ignored.
- Runs on the user's Windows PC (RTX 4060 8 GB). One model on the GPU at a time; stay under 7 GB VRAM.
- Pin `faster-whisper==1.2.1` and `ctranslate2==4.9.1`. No other new Python or npm packages.
- Model B: after loading, set `model.max_length = 880` (never higher). Windows ≤10 s; P4 retry as ≤5 s windows when tokens ≥ cap−2, or window ≥2.5 s with <12 tokens/s; keep retry only if more tokens and compression ratio ≤2.4.
- LLM is used for punctuation only. No translation anywhere in new code.
- Thresholds (defaults): `clip_min_s` 15.0, `clip_max_s` 25.0, `piece_max_s` 10.0, `piece_gap_s` 0.3, `tail_min_s` 5.0, `disagree_cer` 0.25, `tie_margin` 0.05.
- Existing dashboard pages (Jobs, Review, Dataset, Settings) keep every feature. Old datasets' clips, including `translated` and `pending_llm` clips, keep working.
- Default model paths: `MODEL_A_PATH=C:\Users\BS23-DESKTOP-00038\Downloads\qwen-3-asr-finetune-1\safetensors`, `MODEL_B_PATH=C:\Users\BS23-DESKTOP-00038\Downloads\whisper-finetune-1\modelswhisper-v3-bn-version1`. The pipeline never needs the HF Hub for these.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Server restart while a Studio video waits at `awaiting_cut` or is `fetching`.** Expected: `awaiting_cut` stays put (not queued); `fetching` resumes its download. Test in Task 8.
2. **Ranges typed or dragged out of order, overlapping, past the end, or under 1 s.** Expected: normalised on the server (sorted, clamped, merged, tiny ones dropped); `start ≥ end` or all ranges empty after clamping → 422. Tests in Tasks 4 and 10.
3. **LLM returns text with no sentence terminator anywhere (or is down).** Expected: clips still form at 15–25 s, flagged `incomplete_sentence` / `unsplit`, never silently dropped. Test in Task 6.
4. **Model changed in Settings while a video is in `transcribe_a`/`transcribe_b`.** Expected: the running stage finishes with the model it started with and the video records that id; the next run uses the new one. Test in Task 7.
5. **Upload interrupted or the same file dropped twice.** Expected: partial file removed and no video row; second drop → 409 with the existing id. Tests in Task 10.

---

### Task 1: Worktree runtime setup and baseline

**Files:**
- Create: `.env` (copy of the main checkout's `youtube-tts-dataset/.env`, not committed)
- Modify: `.gitignore` (repo root; add `var-studio/`)
- Modify: `.env.example` (add `MODEL_A_PATH`, `MODEL_B_PATH` lines with the defaults, comment: "local folder or HF repo id")
- Modify: `frontend/vite.config.ts` (backend URL from `process.env.YTDS_PORT ?? "8765"`)
- Modify: `backend/pyproject.toml` (`"faster-whisper==1.2.1"`, `"ctranslate2==4.9.1"`)

- [ ] **Step 1:** Copy `C:\Users\BS23-DESKTOP-00038\Projects\BanglaVoiceBotFineTuned\youtube-tts-dataset\.env` into the worktree's `youtube-tts-dataset/.env`; set `YTDS_PORT=8766`, `YTDS_DATA_DIR=youtube-tts-dataset/var-studio`, and add `MODEL_A_PATH` / `MODEL_B_PATH` with the defaults above.
- [ ] **Step 2:** Make the gitignore, `.env.example`, vite and pyproject edits. `run.ps1` already reads `YTDS_PORT` from `.env`; in `-Dev` mode it must export `YTDS_PORT` to the Vite window (check and add if missing).
- [ ] **Step 3:** `uv sync` in `backend`, `npm ci` in `frontend`. Expected: both succeed; `uv.lock` resolves faster-whisper 1.2.1 and ctranslate2 4.9.1 (lock may change only in the pin markers).
- [ ] **Step 4:** Baseline: `uv run pytest -q -m "not gpu"` and `npm test`. Expected: all pass. Record the counts in the commit body.
- [ ] **Step 5:** `git status` shows no `.env` or `var-studio`. Commit: `chore(ytds): worktree runtime on port 8766, pin whisper libs`.

---

### Task 2: Config, thresholds and store schema

**Files:**
- Modify: `backend/ytds/config.py`, `backend/ytds/store.py`
- Test: `backend/tests/test_config.py`, `backend/tests/test_store.py`

**Interfaces:**
- Produces:
  - `Thresholds`: remove `merge_gap_s`; defaults `clip_min_s=15.0`, `clip_max_s=25.0`; new `piece_max_s=10.0`, `piece_gap_s=0.3`, `tail_min_s=5.0`, `disagree_cer=0.25`, `tie_margin=0.05`.
  - `Settings.model_a_path: str`, `Settings.model_b_path: str` (env `MODEL_A_PATH`, `MODEL_B_PATH`; defaults from Global Constraints).
  - `Stage` literal: `download, cut, separate, pieces, transcribe_a, transcribe_b, punctuate_pieces, group, quality, select, punctuate` plus legacy `segment, transcribe, fallback` (kept so old marks still type-check).
  - `VideoStatus` adds `fetching`, `awaiting_cut`. `DropReason` adds `bad_stt`.
  - `VIDEO_COLS` adds `origin, source, cuts, media, model_a, model_b`. `CLIP_COLS` adds `flag, text_pick, score_a, score_b, range_idx`.
  - `Store.add_video(dataset_id, video_id, url, *, origin="dashboard", source="youtube", status="queued") -> bool`.
  - `Store.replace_pieces(dataset_id, video_id, pieces: list[dict]) -> None`, `Store.pieces(dataset_id, video_id) -> list[dict]` (ordered by idx), `Store.update_piece(dataset_id, video_id, idx, **fields) -> None`. Piece columns: `idx, range_idx, start_s, end_s, text_a, text_b, b_tokens, b_compression, b_retry_failed, text, broken, no_punct`.
  - `Store.delete_video` also deletes the video's pieces.
  - Store `__init__` runs `_migrate()`: `ALTER TABLE ... ADD COLUMN` for every missing column; stage-mark migration (§4.8); threshold migration (stored `clip_min_s==2.0` → 15.0, `clip_max_s==15.0` → 25.0, other values kept; a stored `merge_gap_s` key is ignored).

- [ ] **Step 1: Write failing tests**

```python
def test_old_db_gains_columns_and_keeps_rows(tmp_path):
    # create a DB with the 3ec2ce9 schema (copy the old CREATE statements into the test), insert a video and a clip
    ...
    s = Store(path)
    v = s.list_videos(ds)[0]
    assert v["origin"] is None or v["origin"] == "dashboard"
    assert s.get_clip(ds, "v1_0000")["text_a"] == "x"
    s.update_clip(ds, "v1_0000", flag="disagree", text_pick="b", score_a=0.5, score_b=0.9)

def test_stage_marks_migrated(tmp_path):
    # old video with segment and transcribe done
    s = Store(path)
    for st in ("cut", "pieces", "transcribe_a", "transcribe_b", "punctuate_pieces", "group", "select"):
        assert s.stage_done(ds, "v1", st)

def test_threshold_migration_keeps_user_values(tmp_path):
    # stored {"clip_min_s": 2.0, "clip_max_s": 15.0, "dnsmos_min": 3.4, "merge_gap_s": 0.3}
    t = Store(path).get_thresholds()
    assert (t.clip_min_s, t.clip_max_s, t.dnsmos_min) == (15.0, 25.0, 3.4)
    # stored clip_max_s 20.0 stays 20.0

def test_pieces_roundtrip_and_delete(tmp_path):
    s.replace_pieces(ds, "v1", [{"idx": 0, "range_idx": 0, "start_s": 0.0, "end_s": 4.0}])
    s.update_piece(ds, "v1", 0, text_a="ক", broken=0)
    assert s.pieces(ds, "v1")[0]["text_a"] == "ক"
    s.delete_video(ds, "v1")
    assert s.pieces(ds, "v1") == []

def test_add_video_studio(tmp_path):
    assert s.add_video(ds, "local_abc", "file", origin="studio", source="local", status="fetching")
    assert s.list_videos(ds)[0]["status"] == "fetching"
```

In `test_config.py`: `load_settings(env={"MODEL_A_PATH": "x", "MODEL_B_PATH": "y"}, sub_env=tmp_path / "none.env", root_env=tmp_path / "none.env")` gives `model_a_path == "x"`; with no env the defaults are the two Downloads paths; `Thresholds()` has the new defaults and no `merge_gap_s`.

- [ ] **Step 2:** `uv run pytest tests/test_store.py tests/test_config.py -q` → FAIL.
- [ ] **Step 3:** Implement. Migration runs inside `__init__` under the lock; stage migration only inserts marks that are missing.
- [ ] **Step 4:** Same command → PASS. Then full `uv run pytest -q -m "not gpu"`; fix only breakages caused by the renamed/removed threshold (old `segment` tests are deleted in Task 5).
- [ ] **Step 5:** Commit `feat(ytds): store schema for studio, pieces and two-model selection`.

---

### Task 3: Routing rename and model selection scoring

**Files:**
- Modify: `backend/ytds/pipeline/routing.py`
- Create: `backend/ytds/pipeline/select.py`
- Test: `backend/tests/test_routing.py`, `backend/tests/test_select.py`

**Interfaces:**
- Produces:
  - `route(text: str, duration_s: float, th) -> Literal["auto", "broken"]` (was `"fallback"`).
  - `score(text: str, duration_s: float, th, b_stats: dict | None = None) -> float` in [0, 1]. `b_stats` keys: `max_compression: float`, `retry_failed: bool`.
  - `pick(score_a: float, score_b: float, th) -> Literal["a", "b"]`.
  - `ab_cer(text_a: str, text_b: str) -> float` (normalise with `bn_normalize(..., mode="asr")`, strip punctuation as in `punct._PUNCT`; `torchmetrics.functional.char_error_rate(preds=a, target=b)`; both empty → 0.0; one empty → 1.0).
  - `select_clip(piece_rows: list[dict], duration_s: float, th) -> dict` returning `text_a, text_b, score_a, score_b, text_pick, cer, flag, status, drop_reason` (status `auto`, `flagged` or `dropped`; `flag` `disagree` or None; `drop_reason` `bad_stt` or None). Joins piece texts with single spaces.

Score rules (§4.6), exact: broken → 0. Else `1.0 - (1 - bn_ratio) - 0.1 * max(0, longest_repeat - 2) - min(0.3, 0.02 * distance_outside(cps, 8, 20))`, model B additionally `- 0.3` if `max_compression > 2.4`, `- 0.2` if `retry_failed`; clamp to [0, 1]. `cps` = Bangla chars (as in routing `_chars`) / max(duration, 0.1).

- [ ] **Step 1: Write failing tests** — `test_routing.py`: replace every `"fallback"` expectation with `"broken"`. `test_select.py`:

```python
TH = Thresholds()
GOOD = "আমি আজ বাজারে গিয়েছিলাম এবং অনেক কিছু কিনেছি"  # use a real ~12 chars/s sentence for a 3.5 s duration
def test_broken_scores_zero(): assert score("", 3.0, TH) == 0.0
def test_clean_text_scores_one(): assert score(GOOD, 3.5, TH) == pytest.approx(1.0)
def test_repeat_penalty(): ...  # "ক ক ক ক" style → 1.0 - 0.1 * (run - 2), still above 0 when not broken
def test_rate_penalty_capped(): ...  # cps 30 → penalty 0.2; cps 40 → still capped at 0.3 if not broken by max_chars_per_s; use th with max_chars_per_s=100
def test_b_penalties(): assert score(GOOD, 3.5, TH, {"max_compression": 2.5, "retry_failed": True}) == pytest.approx(0.5)
def test_tie_goes_to_b(): assert pick(0.90, 0.87, TH) == "b"; assert pick(0.95, 0.80, TH) == "a"
def test_select_clip_disagree_flag(): # A and B piece texts with CER > 0.25 → flag "disagree", status "flagged"
def test_select_clip_both_broken_dropped(): # → status "dropped", drop_reason "bad_stt", flag None
def test_select_clip_agree_auto(): # → status "auto", text_pick "b" on equal scores
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(ytds): broken routing and A/B transcript scoring`.

---

### Task 4: Cut stage

**Files:**
- Create: `backend/ytds/pipeline/cut.py`
- Test: `backend/tests/test_cut.py`

**Interfaces:**
- Produces:
  - `GAP_S = 2.0`, `MIN_RANGE_S = 1.0`.
  - `class BadCuts(ValueError)`.
  - `normalize_ranges(cuts: list[list[float]] | None, duration_s: float) -> list[tuple[float, float]]` — None/[] → `[(0, duration_s)]`; any `start >= end` → `BadCuts`; sort, clamp to `[0, duration_s]`, merge overlaps/touching, drop ranges `< MIN_RANGE_S`; empty result → `BadCuts`.
  - `cut_audio(audio_wav: Path, out_dir: Path, cuts: list[list[float]] | None) -> list[dict]` — writes `out_dir/audio.wav` (same sample rate, PCM_16) and `out_dir/map.json`; returns the map: `[{range_idx, cut_start_s, cut_end_s, orig_start_s}]`.
  - `to_orig(t: float, cmap: list[dict]) -> float`.
  - `range_spans(cmap) -> list[tuple[int, float, float]]` (range_idx, cut_start_s, cut_end_s).

- [ ] **Step 1: Write failing tests**: `normalize_ranges([[5, 3]], 60)` raises; `[[10, 20], [15, 30], [50, 50.5], [70, 80]]` with duration 60 → `[(10, 30)]` (50–50.5 is under 1 s; 70–80 is past the end); `None` → `[(0.0, 60.0)]`; all ranges beyond duration → raises. `cut_audio` on a synthetic 48 kHz 10 s sine with cuts `[[1, 3], [6, 8]]` → output length `(2 + 2.0 + 2) s`, samples in the gap all zero, map has `cut_start_s` 0.0 and 4.0; `to_orig(5.0, map) == pytest.approx(7.0)`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement with soundfile (read only the needed frames). **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(ytds): cut stage joins keep ranges with a time map`.

---

### Task 5: Pieces stage (replaces segment)

**Files:**
- Create: `backend/ytds/pipeline/pieces.py` (move `_get_vad`, `vad_regions` from `segment.py`)
- Delete: `backend/ytds/pipeline/segment.py`, `backend/tests/test_segment.py`
- Test: `backend/tests/test_pieces.py`

**Interfaces:**
- Consumes: `range_spans` (Task 4).
- Produces:
  - `plan_pieces(regions: list[tuple[float, float]], spans: list[tuple[int, float, float]], audio16k: np.ndarray, max_s: float, gap_s: float) -> list[dict]` → `{idx, range_idx, start_s, end_s}` in cut-timeline seconds.
  - `pieces(vocals_wav: Path, cmap: list[dict], th) -> list[dict]` (load 16 kHz, `vad_regions`, `plan_pieces`).

Algorithm (not determined by the signature): clip each VAD region to its range span (a region never crosses a range). Within a range, merge consecutive regions while the gap between them is `< gap_s` and the merged span stays `≤ max_s`; a gap `≥ gap_s` always ends a piece unless the resulting piece would be < 1.0 s and the merged one fits within `max_s` (avoid one-word pieces). A single region longer than `max_s` is split at the quietest 20 ms frame in the last 2 s before `max_s` (port `_quietest_split` from `stt-playground/whisper_chunking.py`, in seconds), repeatedly, with no overlap.

- [ ] **Step 1: Write failing tests**: regions `[(0,3),(3.1,6),(7,9)]`, one span `(0,0,20)`, `gap_s=0.3`, `max_s=10` → pieces `(0,6)`, `(7,9)`; a 23 s region with a silent frame planted at 9.2 s and 18.6 s → pieces end at ≈9.2 and ≈18.6 (±0.02), all ≤ 10 s; a region crossing a span boundary is clipped into two ranges with different `range_idx`; idx is 0..n-1 in order.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement; delete `segment.py` and its test. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(ytds): VAD pieces up to 10 s replace the segment stage`.

---

### Task 6: Grouping

**Files:**
- Create: `backend/ytds/pipeline/group.py`
- Test: `backend/tests/test_group.py`

**Interfaces:**
- Produces:
  - `ends_sentence(text: str) -> bool` — stripped text ends in `।`, `?`, `!` or `.`.
  - `PAD_S = 0.1`.
  - `group_pieces(pieces: list[dict], th, punctuated: bool) -> list[dict]` — input pieces have `idx, range_idx, start_s, end_s, text, broken`; output clips `{idx, range_idx, start_s, end_s, piece_idxs: list[int], flag: str | None, status: str | None, drop_reason: str | None}` (`status="dropped"`, `drop_reason="length"` only for dropped tails; otherwise status None — set later).

Rules exactly as §4.3 (1–7) with `clip_min_s`, `clip_max_s`, `tail_min_s`; "largest pause in the 15–25 s window" = the piece boundary inside the clip, with clip span between `clip_min_s` and `clip_max_s`, whose gap to the next piece is largest. When `punctuated=False`, no piece ends a sentence: every clip closes at that largest pause with `flag="unsplit"` (§4.4) instead of `incomplete_sentence`. Padding: `start_s - PAD_S`, `end_s + PAD_S`, never into the neighbour clip and never outside the piece's range span (clamp to the first/last piece of the range ± PAD_S but not past the midpoint of a gap to a neighbour clip).

- [ ] **Step 1: Write failing tests** (helper `P(idx, s, e, text, rng=0, broken=0)`):
  - sentence end at 16 s → clip closes there; next clip starts with the next piece;
  - first sentence end at 9 s, next would exceed 25 s with no other end → clip closes at 9 s (5–15 s, `flag is None`);
  - no sentence end at all within 25 s → closes at the largest gap, `flag == "incomplete_sentence"`;
  - `punctuated=False` → every clip `flag == "unsplit"`, durations within 15–25 s where the pieces allow;
  - tail 6 s ending in `।` → kept; tail 4 s → `status == "dropped"`, `drop_reason == "length"`; tail 8 s without terminator → dropped;
  - broken piece in the middle splits clips; its time range is inside no clip;
  - pieces in two ranges never share a clip;
  - adjacent clips never overlap after padding;
  - **Review Focus 3:** LLM text with zero terminators over 80 s of pieces → 3–5 clips, all flagged, none dropped except a short tail.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(ytds): group pieces into 15-25 s sentence-complete clips`.

---

### Task 7: Models — Whisper B windowing, loaders, registry, Settings API

**Files:**
- Create: `backend/ytds/pipeline/whisper_b.py`, `backend/ytds/models.py`
- Modify: `backend/ytds/pipeline/transcribe.py` (keep Qwen helpers; `load_model_a(path: str)`; replace `transcribe_clips` with `transcribe_pieces_a`), `backend/ytds/gpu.py`, `backend/ytds/api/__init__.py`, `backend/ytds/api/settings.py`, `backend/ytds/exporter.py` (`_model_ids`)
- Delete: `backend/ytds/pipeline/fallback.py`, `backend/tests/test_fallback.py`
- Test: `backend/tests/test_whisper_b.py`, `backend/tests/test_models.py`, `backend/tests/test_transcribe.py`, `backend/tests/test_api.py`, `backend/tests/test_gpu.py`

**Interfaces:**
- Produces:
  - `whisper_b.MAX_LENGTH = 880`, `WINDOW_S = 10.0`, `RETRY_WINDOW_S = 5.0`, `OVERLAP_S = 1.0`, `MIN_TOK_PER_S = 12.0`, `MIN_RETRY_AUDIO_S = 2.5`, `MAX_COMPRESSION = 2.4`.
  - `whisper_b.Window(start: int, end: int, overlap_prev: bool = False)`; `plan_windows(audio, regions, window_s) -> tuple[list[Window], int]`; `join_overlap(prev: str, nxt: str, max_words: int = 8) -> str`; `suspicious(n_tokens: int, dur_s: float, token_cap: int) -> str | None` — ported from `stt-playground/whisper_chunking.py` at `origin/main` 7e5510e (copy their unit behaviour; drop logging and monkey-patching).
  - `whisper_b.decode_pieces(pipeline, audio16k: np.ndarray, pieces: list[dict], token_cap: int) -> list[dict]` → per piece `{text, tokens, compression, retry_failed}`; one batched `pipeline.transcribe(audio, clip_timestamps=[...], vad_filter=False, language="bn", without_timestamps=True, batch_size=16, beam_size=5)` for all pieces, then P4 retries per piece.
  - `whisper_b.load_whisper(path: str) -> Callable[[np.ndarray, list[dict]], list[dict]]` — `WhisperModel(path, device="cuda", compute_type="float16")`, `model.max_length = MAX_LENGTH`, wraps `BatchedInferencePipeline`.
  - `transcribe.load_model_a(path: str) -> Transcriber`; `transcribe.transcribe_pieces_a(pieces, vocals_wav, model, on_progress) -> list[str]`.
  - `models.OPTIONS: list[ModelOption]` with ids `qwen3-asr-bn-ft1`, `qwen3-asr-bn-sayedshaun` (slot `a`), `whisper-bn-ft1`, `whisper-bn-mozilla` (slot `b`); labels from §2.2; defaults `qwen3-asr-bn-ft1`, `whisper-bn-ft1`.
  - `models.resolve(option_id: str, settings) -> str` (folder or repo id; ft1 options read `settings.model_a_path`/`model_b_path`; mozilla → `ct2_dir("mozilla-ai/whisper-large-v3-turbo-bn", settings.data_dir)`, converter restored from commit 5c05bfa, `quantization="float16"`).
  - `models.available(option_id, settings) -> tuple[bool, str]` (ok, expected path or repo id). A value that is an existing folder or contains exactly one `/` and no `:`/`\` (repo id) counts as available.
  - `models.current(store, slot: Literal["a","b"]) -> str` (setting `model_a`/`model_b` or default).
  - `models.loader(slot, store, settings) -> Callable[[], Any]` — returns the object the GPU worker caches, with attribute `.option_id`.
  - `GpuWorker.unload(name: str) -> None` (drop only if that name is loaded).
  - GPU loader names: `"model_a"`, `"model_b"` (registered in `api/__init__._default_runner`, built via `models.loader`); `"whisper_translate"` removed.
  - `GET /api/settings` body gains `"transcription": {"a": {"current": id, "options": [{id, label, available, expected}]}, "b": {...}}`; `"models"` loses `translate`, `model_a` becomes the current A label, adds `model_b`; `llm` label stays.
  - `PUT /api/settings` accepts `model_a`, `model_b` (unknown id or wrong slot → 422; unavailable local option → 422 with the expected path); on change calls `runner.gpu.unload("model_a"/"model_b")` if the runner has `gpu`.
  - `exporter._model_ids()` drops `fallback`; the export `models` block lists the distinct `model_a`/`model_b` ids recorded on exported videos with `resolve()` values.

- [ ] **Step 1: Write failing tests**
  - `test_whisper_b.py`: port the playground's expectations: `join_overlap("আমি আজ বাজারে গিয়েছিলাম", "গিয়েছিলাম এবং")` → `"এবং"`; mid-word overlap case (`করতেছিলাম`/`আছিলাম`); `suspicious(222, 9.0, 224) == "token cap"`; `suspicious(50, 9.0, 224)` contains `"tok/s"`; `suspicious(10, 2.0, 224) is None`; `plan_windows` on a 23 s single region → windows ≤ 10 s, `overlap_prev` on all but first. `decode_pieces` with a fake pipeline: piece 0 returns 40 tokens over 8 s (flagged) and the retry returns 120 tokens with compression 1.8 → text replaced, `retry_failed False`; a retry with compression 3.0 → original kept, `retry_failed True`.
  - `test_models.py`: defaults; `available` false for a missing folder with that path returned; `current` reads the setting.
  - `test_api.py`: `GET /api/settings` has `transcription.a.current == "qwen3-asr-bn-ft1"` and four options in total; `PUT {"model_b": "whisper-bn-mozilla"}` → 200 and FakeRunner/fake gpu saw `unload("model_b")`; `PUT {"model_a": "whisper-bn-ft1"}` → 422; `"translate" not in body["models"]`.
  - `test_gpu.py`: `unload("x")` drops when loaded, no-op otherwise.
  - **Review Focus 4** lives in Task 8's runner test (needs the runner).
- [ ] **Step 2:** Run the five test files → FAIL.
- [ ] **Step 3:** Implement. Delete `fallback.py` and `test_fallback.py`; remove its imports. Settings test fixtures that create `Settings(...)` need no model paths (defaults are strings).
- [ ] **Step 4:** Run → PASS; full non-GPU suite passes except runner tests (fixed in Task 8).
- [ ] **Step 5:** Commit `feat(ytds): model A/B registry, Whisper B windowing, settings model choice`.

---

### Task 8: Runner — new stage graph, Studio gate, selection

**Files:**
- Modify: `backend/ytds/pipeline/__init__.py` (STAGES), `backend/ytds/runner.py`
- Test: `backend/tests/test_runner.py`

**Interfaces:**
- Consumes: Tasks 2–7.
- Produces:
  - `STAGES = ("download", "cut", "separate", "pieces", "transcribe_a", "transcribe_b", "punctuate_pieces", "group", "quality", "select", "punctuate")`.
  - `GPU_STAGES = ("separate", "transcribe_a", "transcribe_b", "quality")`.
  - Paths: `cut_dir(data_dir, ds, vid) -> Path` (`<video>/cut`); `separate` reads `cut/audio.wav`. `vocals_path` unchanged.
  - `StageFns` fields: `download, cut, separate, pieces, transcribe_a, transcribe_b, quality, punctuate`; signatures: `cut(audio_wav, out_dir, cuts) -> list[dict]`, `pieces(vocals_wav, cmap, th) -> list[dict]`, `transcribe_a(pieces, vocals_wav, model, on_progress) -> list[str]`, `transcribe_b(pieces, vocals_wav, model, on_progress) -> list[dict]`; `punctuate` unchanged; `segment/transcribe/fallback` removed.
  - `JobRunner.start_video(dataset_id, video_id, cuts: list[list[float]] | None) -> None` — writes `cuts` (JSON), sets `queued`, enqueues.
  - Active statuses: `fetching` (only while `download` is not done), `queued`, `running`. After `download` on an `origin="studio"` video, the runner sets `awaiting_cut` and stops scheduling it. `start()` resets `running`→`queued` only; `awaiting_cut` untouched; `fetching` stays and resumes.
  - `retry()` on a studio video whose next step is `download` sets `fetching`.
  - `_stage_transcribe_a/_b` record `model_a`/`model_b` option ids on the video (from the loaded object's `.option_id`) and write `text_a`/`text_b`/`b_tokens`/`b_compression`/`b_retry_failed` per piece.
  - `_stage_punctuate_pieces`: per piece compute A/B `score` (`b_stats` from the piece's `b_compression`/`b_retry_failed`), `broken = route(A) == "broken" and route(B) == "broken"`, kept text by `pick`; punctuate each non-broken piece's kept text with the existing `punctuate(text, llm)`, in order; store `text` and `no_punct` per piece.
  - `_stage_group`: per range, `punctuated = not any(p["no_punct"] for p in non-broken pieces of the range)`; call `group_pieces` per range with that flag; `replace_clips` with ids `<vid>_<idx:04d>` (idx continuous across ranges), carrying `range_idx`, `flag`, and dropped tails.
  - `_stage_select`: for each non-dropped clip, `select_clip(pieces_of(clip), dur, th)` and save; `bad_stt` clips dropped.
  - `_stage_punctuate`: machine-text check and source text use `text_a if text_pick == "a" else text_b` (legacy clips with `text_pick` NULL use `text_a`).
  - `_finalize`: `WARN_POOR_STT` when more than half of the clips are `drop_reason == "bad_stt"`.
  - `retry_llm`: reruns `punctuate_pieces` + `group` + `quality` + `select` + `punctuate` for videos with any `flag == "unsplit"` clip **only if** no clip of the video has `status in ("approved", "rejected")` or a `text_final` differing from its machine text; then the normal `PUNCT_RETRY` for `no_punct` clips. Fallback retry code removed.

- [ ] **Step 1: Write failing tests** — rewrite `Fakes` for the new `StageFns` (pieces: 12 pieces of 4 s with 0.5 s gaps; transcribe_a/b return texts where every third piece ends with `।` after punctuate; fake punctuate appends `।` to texts containing `"end"`). New/changed tests:
  - `test_runs_all_stages_in_order` → `stages_for("v1") == list(STAGES)`; clips are 15–25 s or valid tails; `video["model_a"] == "fake-a"`.
  - `test_studio_video_stops_at_awaiting_cut`: `add_video(..., origin="studio", status="fetching")` → after idle, status `awaiting_cut`, only `download` ran; `start_video(ds, vid, [[0, 30]])` → runs to `done`; `cut` fake received `[[0, 30]]`.
  - **Review Focus 1** `test_restart_keeps_awaiting_cut_and_resumes_fetching`: one video `awaiting_cut`, one `fetching` with no download mark → new runner `start()` → first stays `awaiting_cut`, second downloads and ends `awaiting_cut`.
  - `test_dashboard_video_runs_straight_through` (origin default): `cut` receives `None`.
  - `test_broken_pieces_and_bad_stt`: pieces where both fakes return `""` → piece `broken`; a clip whose all pieces are English → dropped `bad_stt`; >50 % → video warning contains `"poor STT"`.
  - `test_llm_down_flags_unsplit_and_retry_regroups`: `llm_up=False` → clips `flag == "unsplit"`; set `llm_up=True`, `retry_llm` → flags cleared; with one clip approved first → regroup skipped.
  - **Review Focus 4** `test_model_change_mid_stage_keeps_started_model`: a fake gpu whose `use("model_a")` returns an object with `option_id="fake-a"`; switch the store setting during `transcribe_a` (inside the fake) → video records `"fake-a"`.
  - Keep and adapt the existing stop/delete/retry/failure-isolation tests to the new stage names.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** `uv run pytest -q -m "not gpu"` → all PASS.
- [ ] **Step 5:** Commit `feat(ytds): runner stage graph for pieces, grouping and A/B selection`.

---

### Task 9: Download and local ingest for Studio

**Files:**
- Modify: `backend/ytds/pipeline/download.py`
- Create: `backend/ytds/pipeline/media.py`
- Test: `backend/tests/test_download.py`, `backend/tests/test_media.py`

**Interfaces:**
- Produces:
  - `download(video_id, url, out_dir, on_progress, cookies=None, ydl_factory=None, *, keep_video: bool = False) -> dict` — `keep_video=True`: format `bestvideo[height<=720][vcodec^=avc1]+bestaudio[acodec^=mp4a]/best[height<=720][ext=mp4]/best[height<=720]`, merge to `out_dir/video.mp4`, then ffmpeg extract `audio.wav` (48 kHz mono s16). Returns metadata plus `"media": "download/video.mp4"`.
  - `media.probe(path: Path) -> dict` (`{"duration_s", "has_audio", "vcodec"}` via `ffprobe -v error -show_entries stream=codec_type,codec_name:format=duration -of json`).
  - `media.needs_proxy(path: Path, vcodec: str | None) -> bool` — True unless suffix in `.mp4/.webm` and vcodec in `h264, vp8, vp9, av1`.
  - `media.make_proxy(src: Path, out: Path) -> None` — `ffmpeg -y -i src -vf scale=-2:min(720\,ih) -c:v libx264 -preset veryfast -crf 26 -c:a aac -b:a 128k out`.
  - `media.ingest_local(src: Path, out_dir: Path, on_progress) -> dict` — probe; no audio → `DownloadError("no audio track")`; extract `audio.wav`; proxy if needed (proxy failure → `media=None`, not an error); returns `{"title", "duration_s", "media"}`.
  - `media.local_video_id(path: Path) -> str` — `"local_" + sha1(file bytes)[:11]`, streaming in 1 MB chunks.

- [ ] **Step 1: Write failing tests**: `needs_proxy` table (mkv/h264 → True, mp4/hevc → True, mp4/h264 → False, webm/vp9 → False); `local_video_id` stable and prefixed; `download(..., keep_video=True)` with a fake ydl factory asserts the format string and returns `media == "download/video.mp4"`; `ingest_local` on a synthetic file made with ffmpeg in the test (`-f lavfi -i sine` + `testsrc` 2 s, mkv) produces `audio.wav` and `media/preview.mp4` (skip the test if ffmpeg is missing); a video-only file raises `DownloadError("no audio track")`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(ytds): video download and local ingest with preview proxy`.

---

### Task 10: Studio API, SPA route, review swap

**Files:**
- Create: `backend/ytds/api/studio.py`
- Modify: `backend/ytds/api/__init__.py` (include router; serve `index.html` for `GET /studio`), `backend/ytds/api/review.py` (ClipPatch `text_pick`), `backend/ytds/runner.py` (stage download dispatch: `source == "local"` → `ingest_local` on the uploaded file at `download/upload<ext>`; `origin == "studio"` YouTube → `keep_video=True`), `backend/ytds/api/jobs.py` (`get_dataset` stage list uses new STAGES — already generic)
- Test: `backend/tests/test_studio_api.py`, `backend/tests/test_api.py`

**Interfaces:**
- Consumes: Tasks 2, 4, 8, 9.
- Produces (all under `/api/datasets/{dataset_id}`), as §6:
  - `POST /studio/videos {url}` → 200 `{video_id}`; playlist (more than one entry from `expand_urls`) → 400; duplicate → 409 `{video_id}`. Adds with `origin="studio"`, `status="fetching"`, enqueues.
  - `POST /studio/upload` (multipart `file`) → streams to `<video_dir>/download/upload<suffix>` via a temp name, hashing as it writes (id from the hash, then rename the temp folder to the id); on any exception or client disconnect the temp file is deleted and no row is added; duplicate → 409 `{video_id}` and the new file deleted. Title = original filename (stored via `update_video(title=...)`), `source="local"`.
  - `GET /studio/videos` → list of studio videos: `{id, title, status, error, warning, duration_s, media_ready: bool, cuts, stages: {stage: bool}, clips: {total, kept, flagged, dropped}}`.
  - `GET /videos/{video_id}/media` → `FileResponse` of `media` (Range works via Starlette); 404 when `media` is NULL or missing.
  - `POST /videos/{video_id}/start {cuts}` → 409 unless status `awaiting_cut`; `normalize_ranges` with the video's `duration_s` → `BadCuts` → 422 `{error}`; also 422 when either model `available()` is false (message names the path); else `runner.start_video`.
  - `GET /videos/{video_id}/clips` → every clip of the video (all statuses) with `reason` (existing `review_reason`, plus `"disagree"` when `flag == "disagree"`), `flag`, `orig_start_s`, `orig_end_s` (from `cut/map.json`; equal to `start_s`/`end_s` when no map).
  - `ClipPatch.text_pick: Literal["a", "b"] | None` — when set without `text_final`, the server sets `text_final` to that model's clip text and `text_pick`; status unchanged.
  - `GET /studio` and `GET /studio/` return `frontend/dist/index.html` when dist exists.
  - `api/jobs.py` `CreateDataset` gains `allow_empty: bool = False`; with it, empty `urls` creates the dataset with no videos (200, `added == 0`). Dashboard calls are unchanged.

- [ ] **Step 1: Write failing tests** (TestClient + FakeRunner gaining `start_video`): add URL (monkeypatch `expand_urls`) → row `fetching`/`studio`; playlist → 400; **Review Focus 5:** upload the same bytes twice → second 409 with the same id and only one file on disk; upload that raises mid-stream (monkeypatch the chunk writer to raise after the first chunk) → 500/400 and no row, no file; `/start` on `queued` → 409; `/start` with `[[5, 3]]` → 422; with `[[10, 20], [15, 30]]` → FakeRunner got `[[10, 30]]`-equivalent normalised list; missing model folder → 422; `/media` Range request `bytes=0-9` → 206 and 10 bytes; `/clips` includes dropped clips and `orig_start_s` mapped through a written `map.json`; PATCH `{"text_pick": "a"}` → `text_final == text_a`; `POST /api/datasets {name, urls: "", allow_empty: true}` → 200 with `added == 0`, and without `allow_empty` → 400 as today; `GET /studio` → 200 HTML when a fake dist exists.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Full non-GPU suite → PASS.
- [ ] **Step 5:** Commit `feat(ytds): studio API for add, upload, preview, start and clips`.

---

### Task 11: Frontend types, API client and Settings model dropdowns

**Files:**
- Modify: `frontend/src/types.ts`, `frontend/src/api.ts`, `frontend/src/pages/Settings.tsx`
- Test: `frontend/src/__tests__/Settings.test.tsx`

**Interfaces:**
- Produces:
  - `STAGES` in `types.ts` = the Task 8 list in the same order; `ProgressRow` `CLIP_STAGES` = `transcribe_a`, `transcribe_b`, `punctuate_pieces`, `punctuate` with unit text "pieces" for the first three and "clips" for `punctuate`.
  - `REASON_LABEL` in `ReviewCard.tsx` adds `disagree: "Models disagree"`.
  - Types: `VideoStatus` adds `"fetching" | "awaiting_cut"`; `Thresholds` matches Task 2 (no `merge_gap_s`); `ClipFlag = "incomplete_sentence" | "unsplit" | "disagree"`; `ReviewClip` adds `flag`, `text_pick`, `score_a`, `score_b`, `orig_start_s?`, `orig_end_s?`; `ReviewReason` adds `"disagree"`; `ModelSlotInfo = {current: string; options: {id: string; label: string; available: boolean; expected: string}[]}`; `AppSettings.transcription: {a: ModelSlotInfo; b: ModelSlotInfo}`; `SettingsUpdate` adds `model_a?`, `model_b?`; `StudioVideo` (Task 10 list shape); `ClipPatch` adds `text_pick?`.
  - API: `addStudioUrl(id, url)`, `uploadStudioFile(id, file, onProgress: (sent: number, total: number) => void): Promise<{video_id: string}>` (XHR, `upload.onprogress`), `listStudioVideos(id)`, `studioMediaUrl(id, videoId)`, `startStudioVideo(id, videoId, cuts: [number, number][])`, `getVideoClips(id, videoId)`.
  - Settings page: section **Transcription models** with two `<select>` labelled "Model A" and "Model B"; unavailable options `disabled` with text `(missing: <expected>)`; saving calls `putSettings({model_a|model_b})`; hint text: "Takes effect from the next transcription run. Clips already transcribed keep their text."; threshold hints for the three routing fields say "…, so the clip is dropped." (exact replacements of "transcribed in English and translated by the LLM" / "so the clip is translated instead"); `merge_gap_s` field removed; new threshold fields `piece_max_s`, `piece_gap_s`, `tail_min_s`, `disagree_cer`, `tie_margin`; validation `clip_min_s < clip_max_s` stays and `tail_min_s ≤ clip_min_s`; `MODEL_LABEL` drops `translate`, adds `model_b: "Transcription (model B)"`, renames `model_a` to "Transcription (model A)" and `llm` to "Punctuation".
- [ ] **Step 1: Write failing tests**: render Settings with a mocked `getSettings` → two selects with current values; a disabled option shows the missing path; changing Model B calls `putSettings({model_b: "whisper-bn-mozilla"})`; no text "translated" anywhere on the page.
- [ ] **Step 2:** `npm test` → FAIL. **Step 3:** Implement. **Step 4:** `npm test` and `npm run typecheck` → PASS.
- [ ] **Step 5:** Commit `feat(ytds-ui): model A/B dropdowns and studio API client`.

---

### Task 12: Studio page — add, fetch, cut

**Files:**
- Create: `frontend/src/pages/Studio.tsx`, `frontend/src/components/RangeEditor.tsx`
- Modify: `frontend/src/App.tsx` (page `studio`, label "Studio"; `currentPage()` also returns `studio` when `location.pathname` is `/studio`), `frontend/src/theme.css`
- Test: `frontend/src/__tests__/RangeEditor.test.tsx`, `frontend/src/__tests__/Studio.test.tsx`

**Interfaces:**
- Consumes: Task 11 API.
- Produces:
  - `RangeEditor` props: `{duration: number; playhead: number; ranges: [number, number][]; onChange(r: [number, number][]): void; onSeek(t: number): void}`. Buttons "Mark in", "Mark out" (mark in stores a pending start at the playhead; mark out closes it, swapping if out < in); each range row has start/end number inputs (0.1 s step), "Play" (seek to start), "Delete"; edges draggable on the timeline bar (pointer events, clamp to [0, duration]); shows "Kept: m:ss of m:ss" (merged total).
  - `mergeRanges(r: [number, number][]): [number, number][]` exported (sort + merge overlaps) — the page sends merged ranges to `/start`.
  - Studio page left column: dataset select + **New dataset** form (name, speaker) that calls `createDataset({name, speaker, urls: "", allow_empty: true})` and selects the new dataset.
  - URL box + "Add", drop zone (drag-over highlight; click opens a file input `accept="video/*,audio/*"`), session list of `listStudioVideos` rows with status pill and current stage progress from `useJobEvents`, refreshed on `stage_done` / `video_done` events and every 5 s while any row is `fetching`/`queued`/`running`.
  - Main panel by status: `fetching` → progress bar (upload progress from XHR for the in-flight upload, else download progress events); `awaiting_cut` → `<video controls src={studioMediaUrl}>` (or "Preview unavailable — type the ranges" when `media_ready` is false), `RangeEditor`, **Start processing** button (disabled while saving; shows 422 error text); `queued`/`running` → stage list (STAGES labels) with per-stage done/progress and **Stop** (existing `skipVideo`); `failed` → error + **Retry** (`retryVideo`); `done` → Task 13 panel.
- [ ] **Step 1: Write failing tests**: `mergeRanges([[5,8],[1,3],[7,10]])` → `[[1,3],[5,10]]`; RangeEditor mark in at 2 then mark out at 1 → range `[1,2]`; delete removes; typed end beyond duration clamps; Studio renders the video element for an `awaiting_cut` row and posts merged ranges on Start; a `fetching` row shows a progress bar; dropping a file calls `uploadStudioFile`.
- [ ] **Step 2:** `npm test` → FAIL. **Step 3:** Implement. **Step 4:** `npm test`, `npm run typecheck` → PASS.
- [ ] **Step 5:** Commit `feat(ytds-ui): studio page with upload, preview and range cutting`.

---

### Task 13: Studio review panel, model swap and export

**Files:**
- Create: `frontend/src/components/StudioReview.tsx`
- Modify: `frontend/src/components/ReviewCard.tsx` (optional props), `frontend/src/pages/Studio.tsx`
- Test: `frontend/src/__tests__/StudioReview.test.tsx`, `frontend/src/__tests__/ReviewCard.test.tsx`

**Interfaces:**
- Consumes: `getVideoClips`, `patchClip`, `exportDataset`, `ReviewCard`.
- Produces:
  - `StudioReview` props `{datasetId: string; videoId: string}`; filters: All, Needs attention (`status in flagged|needs_review|pending_llm` or `flag != null`), Approved, Rejected, Dropped (default Needs attention when non-empty, else All); each card shows `orig_start_s–orig_end_s` as `m:ss.s` and a flag badge ("Incomplete sentence", "Not split by LLM", "Models disagree"); dropped clips render read-only with their drop reason ("Broken transcript" for `bad_stt`).
  - `ReviewCard`: when `clip.text_pick` is set (clips made by the new pipeline; dashboard Review and Studio alike), shows model A and model B texts side by side using the existing diff view, marks the kept one ("kept"), and buttons **Use model A** / **Use model B** that call `patchClip(..., {text_pick})` and update the text box with the response's `text_final`. Note under the buttons: "Punctuation is not re-applied to the swapped text." Clips with `text_pick` NULL (old datasets) render exactly as before.
  - **Export dataset** button at the top of the panel: calls `exportDataset(datasetId)`; on 409 `{unreviewed}` shows the same "skip unreviewed" confirm flow as the Dataset page (reuse its dialog component if extractable, otherwise a `confirm()`); shows the export progress/result from WebSocket `export` events.
- [ ] **Step 1: Write failing tests**: filter counts with a fixture of 6 clips (auto, flagged+disagree, dropped bad_stt, approved, rejected, incomplete_sentence); Use model A calls `patchClip` with `{text_pick: "a"}` and the textbox shows the returned text; the dashboard `ReviewCard` test suite still passes unchanged; Export click calls `exportDataset`.
- [ ] **Step 2:** `npm test` → FAIL. **Step 3:** Implement. **Step 4:** `npm test`, `npm run typecheck`, `npm run build` → PASS.
- [ ] **Step 5:** Commit `feat(ytds-ui): studio clip review with model swap and export`.

---

### Task 14: GPU checks, docs and end-to-end verification

**Files:**
- Modify: `backend/tests/test_smoke_gpu.py`, `README.md` (sub-project), `backend/tests/data/smoke_url.txt` (unchanged unless missing)
- Create: `backend/tests/test_whisper_gpu.py`

- [ ] **Step 1:** `test_whisper_gpu.py` (marked `gpu`): load `MODEL_B_PATH`; take an 18 s Bangla clip (concatenate `tests/data/bn_clip.wav` copies with 0.3 s gaps to 18 s); decode one 18 s window with `max_length=448` → exactly 224 tokens; with 880 → under 440 tokens and more than 224.
- [ ] **Step 2:** `test_smoke_gpu.py` (marked `gpu`): run the real app's runner on the smoke URL as a **dashboard** video and on `tests/data/bn_clip.wav`-based local file as a **studio** video (`start_video` with one range); assert every kept clip is 5–25 s, at least half are 15–25 s, each kept clip's `text_final` ends in a terminator or the clip has a `flag`; peak VRAM < 7 GB (`torch.cuda.max_memory_allocated` for torch models; log `nvidia-smi` for the CT2 model). Also print the **comparison check**: for each kept clip, model A on the whole clip vs model A joined piece text, CER and both texts (report only, never asserts).
- [ ] **Step 3:** Run `uv run pytest -m gpu -q -s` on the PC. Expected: PASS; record durations, VRAM and the comparison summary (mean CER, clips where whole-clip is longer by > 10 %) in the commit body.
- [ ] **Step 4:** README: Studio usage, model settings (`MODEL_A_PATH`, `MODEL_B_PATH`, Settings dropdowns), new stage list, clip length rules, translation removed, worktree port note. Root `.env.example` already updated in Task 1.
- [ ] **Step 5:** Manual check: `.\run.ps1 -Rebuild` in the worktree → open `http://127.0.0.1:8766/studio`: add a YouTube URL, drop a local mp4 and an mkv, cut two ranges on one, start, review, use model A on a disagree clip, approve, export; open `#/jobs`, `#/review`, `#/dataset`, `#/settings` and confirm they work as before. Confirm the live server on 8765 still answers.
- [ ] **Step 6:** Full suites: `uv run pytest -q -m "not gpu"`, `npm test`, `npm run typecheck`, `npm run build` → all PASS. Commit `test(ytds): GPU smoke for studio pipeline; docs`.
