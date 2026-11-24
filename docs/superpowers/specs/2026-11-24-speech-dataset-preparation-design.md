# Speech dataset preparation (STT and TTS) — design

Date: 2026-11-24. Parent specs: `2026-11-23-bangla-voice-mlops-design.md` (§4 Data) and `2026-11-23-training-backends-design.md` (§4 prepare contract, §5 `bn_normalize`, §15 splits, §16 CosyVoice 3).

## 1. Goal

Turn raw speech corpora into curated, versioned datasets that the platform ingests and every ASR and TTS backend trains from. The first source is the Kaggle Bengali.AI Speech corpus (about 1200 h of crowdsourced read Bangla, also published as OOD-Speech). The second is a single-speaker voice set recorded by a colleague (about 1000 min).

Two principles:

- **Curate once, adapt per model.** Curation produces model-independent datasets in the platform's upload layout. Model-specific conversion (tokenization, features, sample rate for the model, speech tokens) stays in each backend's `prepare` step (training-backends §4). No curated dataset is ever shaped for one model.
- **STT and TTS need different data.** STT learns to ignore noise, speakers and channel. TTS learns to reproduce them. The same source therefore yields different datasets with different filters, never one dataset with reversed labels.

## 2. Scope

### 2.1 In scope

- A standalone curation CLI, `tools/dataprep/`, run on the Linux VM (§4).
- The filter profiles and their initial thresholds (§6).
- The curated datasets to produce (§3).
- Text handling for curation, on top of the existing `bn_normalize` (§7).
- TTS audio processing (§8).
- Requirements for the stage-B voice set and its QA (§9).
- Evaluation sets (§10).
- A local-path ingest mode in the platform (§11).
- The CosyVoice 3 tokenizer round-trip gate (§12).
- Corrections to training-backends §16 from the CosyVoice 3 upstream check (§13).

### 2.2 Out of scope

- Train-time augmentation. It belongs to the ASR backends; requirements are listed in §14.
- Running curation as platform jobs with a UI. That is a possible later sub-project; the CLI's step boundaries (§5) are designed so each step can later become a job.
- TTS models without a trainable backend (training-backends §2.2), except for notes that affect data.

## 3. Datasets produced

| Dataset | Task | Source | Profile (§6) | Purpose |
|---|---|---|---|---|
| `bengaliai-asr` v1 | `asr` | Bengali.AI | `asr-v1` | STT training, round 1 |
| `bengaliai-asr` v2 | `asr` | Bengali.AI | `asr-v2` | STT training after the CER filter |
| `bengaliai-tts-strict` | `tts` | Bengali.AI | `tts-strict` | Bangla pretraining for F5 (from scratch) and other acoustic-sensitive models |
| `bengaliai-tts-loose` | `tts` | Bengali.AI | `tts-loose` | Bangla pretraining for the CosyVoice 3 LLM stage |
| `voice-<name>` | `tts` | Colleague's recordings | `voice` | Final single-speaker voice (stage B) |
| `domain-eval-asr` | `asr`, eval-only | Recorded through the app | none | ASR test on real voicebot audio |
| `domain-eval-tts` | text list | Written by hand | none | Held-out TTS sentences |

TTS training runs in two stages:

- **Stage A (Bangla pretraining):** a large multi-speaker set teaches the model Bangla pronunciation. `tts-strict` or `tts-loose`, by model.
- **Stage B (voice):** the clean single-speaker set fixes the voice and audio quality. A warm-start run (training-backends §14) continues from the stage-A checkpoint.

TTS model lineup this data serves (2026-11-24):

| Model | Role | Stage A | Stage B |
|---|---|---|---|
| CosyVoice 3 (0.5B) | Main | `tts-loose`, LLM only | `voice-<name>` |
| Qwen3-TTS 0.6B Base | Second | None (official recipe is single-speaker) | `voice-<name>` |
| F5-TTS from scratch | Third | `tts-strict` | `voice-<name>` |
| IndicF5, XTTS v2 | Benchmark only (non-commercial weights) | — | — |

ASR models: Whisper `large-v3-turbo` and Qwen3-ASR-1.7B are the main tracks. BanglaConformer (`bengaliAI/BanglaConformer`, NeMo Conformer-CTC, trained on the same corpus) is a baseline and a fast labeller for the first CER pass (§6.2).

## 4. Curation CLI

### 4.1 Layout

```
tools/dataprep/
  pyproject.toml, uv.lock      own venv (torch, torchmetrics[audio], onnxruntime-gpu,
                               librosa, soundfile, pyloudnorm, pandas, pyarrow,
                               speechbrain or wespeaker, brouhaha, bnunicodenormalizer)
  dataprep/__main__.py         python -m dataprep <command> ...
  dataprep/sources/bengaliai.py
  dataprep/steps/*.py          one module per step (§5)
  profiles/*.yaml              filter profiles (§6), committed
  tests/                       CPU tests on tiny fixtures
```

- It does not import `app.*` heavy modules. It may import `app.core.text` (pure Python) so curation and training use the same `bn_normalize`.
- Its lockfile must install on the Linux VM (RTX 5090, CUDA 12.8+).
- GPU steps with conflicting pins live in extras. The `cosyvoice` extra follows upstream CosyVoice's pins (`transformers==4.51.3`, `numpy<2`), except torch `2.7.1+cu128`.

### 4.2 Work directory

All intermediate state for one source lives in one work directory, by default `var/dataprep/<source>/`:

```
var/dataprep/bengaliai/
  clips.parquet          one row per source clip: id, src_path, src_sr, text_raw, src_split
  text.parquet           step outputs, one file per step, keyed by id
  audio_stats.parquet
  dnsmos.parquet
  brouhaha.parquet
  cer_<labeller>.parquet
  speakers.parquet
  selections/<profile>.parquet
  exports/<dataset>/     the platform layout (§4.4)
```

- Every step reads `clips.parquet` plus the step files it needs, and writes only its own file.
- Every step is resumable. Rerunning skips ids already present. `--force` recomputes.
- Every step file records the step version and parameters in its Parquet metadata, so a stale file is detected.

### 4.3 Commands

| Command | Does | Hardware |
|---|---|---|
| `import bengaliai --src <dir>` | Read `train.csv` and `train_mp3s/`; write `clips.parquet` with the source split | CPU |
| `text` | Clean text (§7.1), flags, char-per-second | CPU |
| `audio-stats` | Duration, source sample rate, clipping share, silence share, LUFS | CPU |
| `dnsmos` | DNSMOS P.835 scores (§6.1) | CPU or GPU |
| `brouhaha` | SNR, C50, speech-activity segments | GPU |
| `transcribe --labeller <id>` | Transcribe and score CER against `text_clean` | GPU |
| `speakers` | Speaker embeddings and pseudo-speaker clusters (§8.2) | GPU |
| `select --profile <name>` | Apply a profile; write the selection and a report | CPU |
| `listen --profile <name>` | Write a listening sheet: 10 clips per score band, as a static HTML page with players | CPU |
| `export --profile <name> --dataset <name>` | Write the platform layout (§4.4), with audio processing for TTS (§8) | CPU |
| `check-voice --src <dir>` | QA report for stage-B recordings (§9.3) | GPU |
| `roundtrip-cosyvoice run` / `finalize` | The CosyVoice 3 gate (§12) | GPU |

On the VM the competition files are already downloaded to the kagglehub cache: `/home/xr23/.cache/kagglehub/competitions/bengaliai-speech` (`train.csv`, `train_mp3s/`). `import bengaliai` reads them in place and stores absolute `src_path` values, so nothing is copied. Keep that cache until the TTS exports are ingested: `audio-stats`, `dnsmos`, `brouhaha` and the TTS export read the original MP3s from it.

### 4.4 Export layout

Exports follow the platform's upload layout (parent spec §4.2, training-backends §15.2) so they ingest unchanged:

- `asr`: `metadata.csv` + `audio/`, columns `file`, `text`, `split`, plus optional `speaker`, `channel`, `noise`.
- `tts`: `metadata.csv` + `wavs/`, columns `file`, `text`, `speaker`, `split`, plus optional `style`.
- `curation.json` beside `metadata.csv`: source, profile name and full contents, step versions, counts and hours removed per filter, the git commit of `tools/dataprep`, and the provenance fields to enter at ingest.

`text` is the **cleaned** text (§7.1), not a `bn_normalize` output. Each backend applies `bn_normalize` in the mode it needs at prepare time, so a normalizer change never requires re-export.

## 5. Pipeline order

### 5.1 STT

1. Check the Bengali.AI license (§15). Until it is confirmed, ingest with `commercial_ok: false`.
2. `import`, `text`, `audio-stats`.
3. `select --profile asr-v1`. Review the character report (§7.2) and fix rules until no unexpected character remains.
4. `export` as `bengaliai-asr` and ingest it (§11).
5. Train round 1 on the platform (Whisper and Qwen3-ASR).
6. `transcribe` with the best round-1 model.
7. `select --profile asr-v2`, `export`, ingest. This becomes `bengaliai-asr` v2.
8. Retrain and evaluate on `val` plus `domain-eval-asr`.

The first CER pass can use BanglaConformer instead of waiting for round 1 (§6.2).

### 5.2 TTS stage A

1. Reuse the STT work directory: `text`, `audio-stats` and the CER scores are shared.
2. `dnsmos`, then `brouhaha` on the clips that pass the DNSMOS floor of the loosest profile.
3. `speakers`.
4. `listen` per profile. Set the final thresholds by listening, then commit them to the profile.
5. `select` and `export` `bengaliai-tts-strict` and `bengaliai-tts-loose`. Export decodes the **original MP3s**, never the 16 kHz ASR audio (§8.1).

### 5.3 TTS stage B

1. The colleague delivers recordings to the spec in §9.1.
2. `check-voice`. Every flagged take is re-recorded or dropped.
3. `export --profile voice` and ingest.

### 5.4 CosyVoice 3 gate

Run `roundtrip-cosyvoice` (§12) as soon as some clean Bangla audio exists. Use stage-B audio if available, otherwise the top DNSMOS clips. Run it **before** any CosyVoice 3 training.

## 6. Filter profiles

### 6.1 Scores

| Score | Source | Meaning |
|---|---|---|
| `dur` | `audio-stats` | Seconds |
| `clip_frac` | `audio-stats` | Share of samples at or above 0.99 full scale |
| `silence_share` | `audio-stats` | Share of 20 ms frames below −50 dBFS |
| `cps` | `text` | Cleaned characters per second |
| `dnsmos_ovrl`, `dnsmos_sig`, `dnsmos_bak` | `dnsmos` | 1–5; higher is better. `torchmetrics` `deep_noise_suppression_mean_opinion_score`, 16 kHz, `personalized=False` |
| `snr_db`, `c50_db` | `brouhaha` | Mean over speech frames |
| `max_pause_s` | `brouhaha` | Longest internal non-speech gap |
| `cer_<labeller>` | `transcribe` | CER of the labeller's output against the cleaned text, both through `bn_normalize(mode="asr")` |
| `has_latin`, `has_digits` | `text` | Flags |

### 6.2 Labellers

A labeller is any ASR model that `transcribe` can load. It is named in the score column, so scores from several labellers coexist.

- `bangla-conformer`: available now, fast (CTC). It trained on this corpus, so it may have memorized wrong labels and score them as correct. Use it only with a high threshold.
- `round1-<run-id>`: the best round-1 model from the platform, loaded from its exported model directory.

### 6.3 Profiles

Initial values. `tts-*` thresholds are final only after the listening pass (§5.2 step 4).

| Filter | `asr-v1` | `asr-v2` | `tts-loose` | `tts-strict` | `voice` |
|---|---|---|---|---|---|
| `dur` (s) | 0.5–30 | 0.5–30 | 1–15 | 1–15 | 1–20 |
| `clip_frac` | — | — | < 0.001 | < 0.001 | < 0.001 |
| `cps` | within p0.5–p99.5 | within p0.5–p99.5 | within p1–p99 | within p1–p99 | — |
| CER | — | ≤ 0.5 | ≤ 0.05 | ≤ 0.05 | ≤ 0.02 (flag, then review) |
| `dnsmos_ovrl` | — | — | ≥ 2.6 | ≥ 3.2 | ≥ 3.5 (flag) |
| `dnsmos_bak` | — | — | ≥ 3.0 | ≥ 3.8 | — |
| `snr_db` | — | — | — | ≥ 25 | ≥ 35 (flag) |
| `c50_db` | — | — | — | ≥ 30 | — |
| `max_pause_s` | — | — | ≤ 1.5 | ≤ 1.0 | ≤ 1.0 (flag) |
| `has_latin` | keep | keep | drop | drop | keep |
| Speaker cluster min. audio | — | — | 5 min | 10 min | — |
| Expected size | ~1150 h | ~1000–1100 h | 800–1000 h | 150–400 h | ~16 h |

Why the two TTS profiles differ:

- **`tts-loose` (CosyVoice 3):** only the LLM trains, mapping text to speech tokens. The speech tokenizer was trained for recognition, so its tokens tolerate noise; the frozen flow model and vocoder produce the acoustics. Upstream guidance for a new language is "at least 1000 hours" (CosyVoice issue #1336), so volume matters more than polish. The label filter stays strict: wrong text is harmful at any volume.
- **`tts-strict` (F5 and acoustic models):** the model reproduces the audio directly, so noise, reverb and pauses must go.

### 6.4 Selection report

`select` writes `selections/<profile>.report.json`: rows and hours before and after each filter, in order, and the rows removed by each. `export` copies it into `curation.json`.

## 7. Text

### 7.1 Cleaning (stored `text`)

Applied by the `text` step to every source:

1. `bnunicodenormalizer` (Bengali.AI's normalizer). It repairs broken vowel-sign and conjunct sequences that NFC alone does not.
2. Unicode NFC.
3. Remove zero-width characters except ZWJ/ZWNJ inside conjuncts (same rule as `bn_normalize`).
4. Unify dandas and quotes; collapse whitespace; strip the ends.
5. Keep punctuation, digits and Latin text. Removing them is a normalizer decision (`bn_normalize`), not a curation decision.

The cleaned text must round-trip: `bn_normalize(clean(t), mode=m) == bn_normalize(t, mode=m)` for all test fixtures. This stops curation from changing what the backends see, apart from repairs.

### 7.2 Character audit

`select` writes the character inventory of the selected rows: every code point, its count and five example rows. Any character outside the Bengali block, ASCII letters, digits and the punctuation set is listed as unexpected. Each unexpected character gets a fix in §7.1 or a drop rule, so the audit is empty before export.

### 7.3 Train/val text overlap

Rows in `train` whose `bn_normalize(mode="asr")` text equals a `val` row's text are dropped from `train`. The same sentence read by another speaker otherwise inflates validation scores.

### 7.4 TTS text requirements

- At runtime the TTS input is raw LLM output, which `bn_normalize(mode="tts")` verbalizes. Training text goes through the same function in the backend's prepare step, so training and inference match.
- For stage B, the transcript must say exactly what the speaker said. Where the speaker read a number as words, the transcript holds words. `check-voice` (§9.3) catches mismatches.
- Pronunciation fixes for TTS go into an override lexicon used by `bn_normalize(mode="tts")`. No G2P is used: all TTS models in §3 take graphemes or byte-level BPE. The lexicon is a later addition to `app.core.text`, out of scope here.

## 8. TTS audio processing

### 8.1 Source audio

- TTS exports decode the original MP3s. The `import` step records each clip's source sample rate.
- Export never upsamples a clip to claim more bandwidth than its source has. A clip whose source rate is below 24 kHz is exported at 24 kHz but flagged `band_limited` in `curation.json` counts.

### 8.2 Pseudo-speakers

Bengali.AI has no speaker IDs, and the `tts` layout requires `speaker`.

1. Embed each candidate clip (ECAPA-TDNN, 192-d).
2. Cluster with agglomerative clustering on cosine distance; the threshold is a step parameter.
3. Name clusters `bai-<n>`. Drop clusters below the profile's minimum audio (§6.3).

F5 ignores speaker labels in training but uses them to pick reference clips. CosyVoice 3 uses them for speaker embeddings.

### 8.3 Export processing (all `tts` exports)

Applied in this order:

1. Decode, convert to mono.
2. Trim leading and trailing non-speech using the brouhaha segments, keeping 150 ms of padding.
3. Loudness-normalize to −23 LUFS integrated, then limit true peak to −1 dBTP (`pyloudnorm`). This is inside the ingest validator's −30 to −14 LUFS window.
4. Resample to 24 kHz.
5. Write 16-bit PCM WAV.

24 kHz is the native rate of every TTS model in §3. Each backend resamples further if it needs to (CosyVoice 3 uses 16 kHz for its tokenizer).

ASR exports are decoded, converted to mono, resampled to 16 kHz and written as FLAC. They are not trimmed or loudness-normalized.

## 9. Stage-B voice set

A colleague is recording about 1000 minutes (about 16.7 h) of one speaker.

### 9.1 Recording requirements (handed to the colleague)

| Item | Requirement |
|---|---|
| Format | WAV, 48 kHz, 24-bit, mono, one file per utterance |
| Room | Quiet and treated; no fans, traffic or echo |
| Setup | Same microphone, distance and gain in every session; note any change |
| Style | Friendly assistant, conversational; not news-reading. The same style in every session |
| Utterance length | 2–15 s; one sentence or a short paragraph |
| Takes | Re-record any take with a misread word, a cough, a click or a long pause |
| Transcript | Exactly what was said. Numbers, dates and prices written the way they were spoken |
| Delivery | `metadata.csv` (`file`, `text`) and `wavs/`, plus a short session log (date, mic, notes) |
| Consent | Written consent and a contract covering commercial use of the voice and models trained on it |

### 9.2 Script coverage

The recording script should cover:

- **Domain:** product names, prices, order numbers, dates, delivery addresses, greetings, apologies, confirmations.
- **Sentence types:** statements, yes/no and wh-questions, lists, short replies ("জি", "ঠিক আছে").
- **Code-mixing:** Bangla with English product terms, written the way the LLM will write them.
- **Phonetic coverage:** every vowel sign, every common conjunct, and aspirated, unaspirated and retroflex contrasts, each at least 20 times. `check-voice` reports coverage against this list.

### 9.3 `check-voice`

Per take:
- CER of an STT transcript against the given text. Flag anything above 0.02; a person reviews every flag.
- DNSMOS, SNR, clipping, LUFS, longest pause, sample rate and bit depth. Flag values outside the `voice` profile.

Per set:
- Total hours, phonetic coverage (§9.2), and speech-rate drift across sessions.

Output: an HTML report with a player per flagged take. Flagged takes are re-recorded or dropped before export.

### 9.4 Split

- 150 takes, chosen to cover the domain categories, are held out as `test` and never used for training.
- The rest go to `train` and `val` through the platform's default ratios.

## 10. Evaluation sets

### 10.1 `domain-eval-asr`

- 1–2 h of conversational voicebot queries: e-commerce intents, product names, numbers and prices, with the accents of expected users.
- Recorded **through the real app path** (browser microphone, the app's transport, captured at the backend), so browser noise suppression, auto gain and the codec are all present. The app's `getUserMedia` settings are fixed before recording and not changed afterwards without re-recording.
- Mostly short turns (1–3 s), as produced by VAD.
- Ingested with the eval-only flag (parent spec §4.4).

### 10.2 `domain-eval-tts`

- 150–200 sentences of domain text written for the voicebot, none present in any training transcript (checked after `bn_normalize(mode="asr")`).
- Includes numbers, prices, dates, code-mixed terms and questions.
- Stored as a UTF-8 text file, one sentence per line, under `var/dataprep/eval/`.
- Metrics, when the TTS eval sub-project adds them: CER of a fixed STT model on the synthesized audio, predicted MOS (UTMOS), speaker similarity to the stage-B voice, and human listening.

### 10.3 ASR scoring rule

WER and CER always use `bn_normalize(mode="asr")` on both texts. Whisper's `BasicTextNormalizer` is never used for Bangla: it replaces Unicode category `M` characters with spaces, and Bangla vowel signs are category `M`.

## 11. Local-path ingest

The curated ASR dataset is 60–140 GB. Uploading it to the same machine it already sits on wastes time and disk.

### 11.1 API

`POST /datasets/ingest-local` with `{path, task, name, provenance}`. It runs the same ingest job as an upload, with two differences:

- **Source:** a directory, not a zip. The directory must resolve (after following symlinks) to a location inside `settings.import_root`, default `var/import/`. Any other path is rejected with `403`. `..` segments and symlinks that leave the root are rejected.
- **File transfer:** files are hard-linked into the version directory when the source and `data/` are on one filesystem, otherwise copied. The source directory is never modified or deleted.

Everything else is unchanged: `validate_audio`, provenance fields, the version hash, the split rules (training-backends §15) and the disk check.

### 11.2 Validation at scale

`validate_audio` decodes every file. At about 1M clips this takes hours on one core.
- Validate in a process pool with progress reported to the job.
- Store at most 1000 warnings per warning code, plus the total count per code. The report shows the counts and a sample.

### 11.3 UI

The dataset upload page gains a "From server path" option: a path field, the task, and the same provenance form. It lists the directories under `import_root` that contain a `metadata.csv` or `metadata_<split>.csv`.

### 11.4 Tests

- Path outside the root, path with `..`, symlink escaping the root: rejected.
- Same-filesystem source: version files are hard links; the source is unchanged.
- Warning capping: counts are exact, stored samples are capped.

## 12. CosyVoice 3 round-trip gate

### 12.1 Question

`speech_tokenizer_v3` has never seen Bangla. If it cannot represent Bangla sounds, then no LLM training fixes it, because the LLM can only predict tokens the tokenizer offers.

### 12.2 Procedure (`roundtrip-cosyvoice`)

1. Take 30 clean Bangla clips: stage-B takes if available, otherwise the top-DNSMOS `tts-strict` clips. Clips must contain aspirated (খ ঘ ছ ঝ ঠ ঢ থ ধ ফ ভ) and retroflex (ট ঠ ড ঢ ণ ড়) consonants.
2. For each clip, extract speech tokens with `speech_tokenizer_v3.onnx`.
3. Resynthesize audio from those tokens with the frozen flow model and vocoder (upstream `CosyVoice3.inference_vc`; no LLM runs). One separate prompt clip supplies the timbre for all clips. The clip itself is never its own prompt: the flow model could then copy content from the prompt mel and hide token losses.
4. Transcribe the original and the resynthesized audio with the same STT model and compute both CERs against the text.
5. Write a report: per-clip CERs, an HTML page with original and resynthesized audio side by side, and a field for each clip's listening verdict.
6. `roundtrip-cosyvoice finalize` combines the verdicts downloaded from the page with the CER result into `gate.json`, and exits non-zero on failure.

Setup: `tools/dataprep/scripts/setup_cosyvoice.sh` syncs the `cosyvoice` extra of the dataprep venv, clones upstream CosyVoice (with the Matcha-TTS submodule) into `var/vendor/CosyVoice`, and downloads the weights into `var/models/Fun-CosyVoice3-0.5B`. The resolved upstream commit is recorded in the run's `summary.json`.

### 12.3 Pass criteria

- Mean CER of resynthesized audio exceeds mean CER of the originals by at most 5 percentage points.
- A listener marks aspirated and retroflex consonants as preserved in at least 27 of 30 clips.

Fail: CosyVoice 3 is dropped as the main track, and F5 from scratch takes its place. Record the decision and the report in the dataset notes.

## 13. Corrections to training-backends §16 (CosyVoice 3)

Upstream check of `FunAudioLLM/CosyVoice` and `Fun-CosyVoice3-0.5B-2512` (2026-11-24):

| Item | Finding | Change |
|---|---|---|
| License | Code Apache-2.0 (LICENSE file). Weights Apache-2.0 per the HF metadata; no LICENSE file in the HF repo | None; `commercial_ok: true` stands |
| Trainable components | `examples/libritts/cosyvoice3/run.sh` prints "We only support llm traning for now" although it loops over `llm flow hifigan` | §16.4: default `components` becomes `["llm"]`. `flow` stays allowed but is marked unsupported upstream |
| Text prefix | `prepare_data.py` is called with `--instruct "You are a helpful assistant.<|endofprompt|>"`, prepended to each text in CosyVoice 3 | §16.3: the `cosyvoice` converter adds this prefix |
| Speaker embedding | `cosyvoice3.yaml`: `use_spk_embedding: False # change to True during sft` | §16.4: set `True` for stage B; leave `False` for stage A |
| Checkpoint averaging | Stage 6 reads `exp/cosyvoice/...`, stage 5 writes `exp/cosyvoice3/...` | Note for the in-process trainer port; do not copy the path |
| `transformers` version | Issue #1953: CosyVoice 2/3 output collapses on `transformers >= 4.53` | §16.2: pin `transformers < 4.53` until the issue's fix is confirmed |
| Data volume | Issue #1336 (maintainer): a new language needs "at least 1000hours data" | §16.1 note updated; stage A uses `tts-loose` (§6.3) |
| Bangla tokenizer | Untested | §12 gate before any training |

## 14. Requirements handed to the ASR backends

Not built here. Recorded so the ASR backends implement them:

- **Channel:** app/web audio, not telephony. No 8 kHz downsampling and no phone-codec simulation.
- **Noise:** add background noise at 10–30 dB SNR to 20–30 % of training clips.
- **Reverb:** mild simulated room reverb on a small share of clips.
- **Codec:** encode and decode a share of clips with Opus at 16–32 kbps only if the app's transport is WebRTC. Skip it for raw PCM over WebSocket.
- **Token length:** Whisper and Qwen3-ASR spend many byte-BPE tokens per Bangla word. Skip rows over the label limit (448 for Whisper), as training-backends §4.4 already does.
- **Qwen3-ASR language label:** Bengali is not a supported language. The converter uses one fixed language label for every row.

## 15. Licensing and provenance

- **Bengali.AI Speech:** the Kaggle competition rules and data license are not yet checked. Ingest with `commercial_ok: false` until they are; any run on it then carries a non-commercial badge (parent spec §4.6).
- **Stage-B voice:** `commercial_ok: true` only after the consent and contract in §9.1 are signed.
- **BanglaConformer:** HF metadata says MIT, with no LICENSE file. Used only as a labeller and baseline, never shipped.
- **bnunicodenormalizer:** check its license before adding it to `tools/dataprep`.
- `curation.json` records the source, profile and tool commit, so each dataset version is traceable to the curation run that made it.

## 16. Testing

- `tools/dataprep` CPU tests on a fixture of about 20 short clips with known texts:
  - `text` cleaning: table of inputs and expected outputs, including broken vowel signs, `য়` variants, ZWJ/ZWNJ and stray Latin.
  - The round-trip property in §7.1.
  - `select`: each filter removes exactly the expected rows; the report counts match.
  - `export`: the output passes the platform's `validate_audio` with no errors; TTS audio is 24 kHz mono and within −23 ± 1 LUFS.
  - Resume: rerunning a step with existing output computes nothing.
- Scorers (DNSMOS, brouhaha, speakers, transcribe) are tested on the VM with a smoke script on the same fixture, not in CI.
- Local-path ingest tests (§11.4) live in `backend/tests`.

## 17. Delivery order

1. `tools/dataprep` scaffold: `import`, `text`, `audio-stats`, `select`, `export` for ASR.
2. Local-path ingest (§11). Ingest `bengaliai-asr` v1 and start STT round 1.
3. `roundtrip-cosyvoice` (§12) on top-DNSMOS clips. Its result decides the TTS main track.
4. `dnsmos`, `brouhaha`, `speakers`, `listen`. Tune and export `tts-strict` and `tts-loose`.
5. `transcribe` with BanglaConformer, then with the round-1 model. Export `bengaliai-asr` v2.
6. `check-voice` and the stage-B export when the recordings arrive.
7. `domain-eval-asr` recording and `domain-eval-tts` sentences.
8. Apply the §13 corrections to training-backends §16.

## 18. Open items

- Bengali.AI license and terms (§15).
- `bnunicodenormalizer` license.
- The app's transport (WebRTC or WebSocket PCM). It decides the Opus augmentation (§14).
- The `channel` field is documented as `studio`/`phone` but not validated at ingest, so app audio uses `channel=web` with no code change. Update the parent spec's §4.2 table.
- Whether Bengali.AI audio has a usable source sample rate for TTS (§8.1). Check it during `import`; if most clips are band-limited, `tts-strict` shrinks.
- Qwen3-TTS: which language setting works best for Bangla in its fine-tune. Decide by a small experiment on stage B.
