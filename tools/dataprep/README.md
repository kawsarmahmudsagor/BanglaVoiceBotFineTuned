# dataprep

Curation CLI that turns Bangla speech corpora (the Kaggle Bengali.AI set and a single-speaker recorded voice set) into ASR and TTS datasets for the training platform. Design: `docs/superpowers/specs/2026-11-24-speech-dataset-preparation-design.md`; section numbers below refer to it.

Everything runs on the Linux VM (Ubuntu, RTX 5090, CUDA 12.8+). Run all commands from the repo root.

## Setup

The dataprep venv is managed with uv and has one extra per heavy dependency set:

- `score`: DNSMOS, Brouhaha, speaker embeddings (`dnsmos`, `brouhaha`, `speakers`).
- `labeller`: Whisper via transformers and NeMo (`transcribe`, `check-voice`).
- `cosyvoice`: the CosyVoice 3 round-trip gate. It pins `transformers==4.51.3` and conflicts with `labeller`, so never sync both.

```bash
uv sync --project tools/dataprep --extra score --extra labeller   # score and labeller coexist
export HF_TOKEN=hf_...   # required: the Brouhaha / pyannote weights are gated on Hugging Face
alias dp='uv run --project tools/dataprep --no-sync python -m dataprep'
```

All commands take `--work <dir>` (default `var/dataprep/bengaliai`, or `var/dataprep/voice` for `check-voice`); step files in it are versioned and rerun with `--force` (all rows) or `--retry-errors`.

## Rules to keep in mind

- Start from the kagglehub cache, `/home/<user>/.cache/kagglehub/competitions/bengaliai-speech` (`$BENGALIAI_SRC`). The import reads it in place and records absolute paths, so **do not delete that cache before the TTS exports are ingested**.
- Write exports under the platform's `var/import/`, on the same filesystem as its dataset store, so ingest can hard-link instead of copying.
- Ingested hard-linked files become read-only (0444). Replace a file (write `<name>.tmp-<pid>`, then `os.replace`), never edit it in place.
- `brouhaha` must run before `select` for the TTS profiles and before the `voice` select; those profiles filter on `speech_span_s`, `snr_db`, `c50_db` and `max_pause_s`, and `select` stops with a message naming the missing step.
- Bengali.AI exports carry `commercial_ok: false` until its license is checked.
- CER and WER always use `bn_normalize(mode="asr")` on both sides; curated `text` is the cleaned transcript.

## 1. STT dataset (spec 5.1)

```bash
dp import bengaliai --work var/dataprep/bengaliai            # --src defaults to $BENGALIAI_SRC, else the kagglehub path
dp text --work var/dataprep/bengaliai --workers 8
dp audio-stats --work var/dataprep/bengaliai --workers 8
dp select --work var/dataprep/bengaliai --profile asr-v1     # prints the per-filter report
dp export --work var/dataprep/bengaliai --profile asr-v1 --dataset asr-v1 --out var/import/bengaliai-asr-v1

# asr-v2 adds a CER filter against a labeller
dp transcribe --work var/dataprep/bengaliai --labeller whisper-hf:openai/whisper-large-v3-turbo
dp select --work var/dataprep/bengaliai --profile asr-v2 --labeller whisper-hf:openai/whisper-large-v3-turbo
dp export --work var/dataprep/bengaliai --profile asr-v2 --dataset asr-v2 --out var/import/bengaliai-asr-v2
```

## 2. TTS dataset, stage A (spec 5.2)

Run the cheap filters first and score only what they keep (`--only-profile`), then select and export.

```bash
dp transcribe --work var/dataprep/bengaliai --labeller whisper-hf:openai/whisper-large-v3-turbo --only-profile tts-loose
dp dnsmos     --work var/dataprep/bengaliai
dp brouhaha   --work var/dataprep/bengaliai --only-profile tts-loose --labeller whisper-hf:openai/whisper-large-v3-turbo
dp speakers   --work var/dataprep/bengaliai --only-profile tts-loose --labeller whisper-hf:openai/whisper-large-v3-turbo
dp listen     --work var/dataprep/bengaliai --profile tts-loose --labeller whisper-hf:openai/whisper-large-v3-turbo   # HTML sheet per DNSMOS band
dp select     --work var/dataprep/bengaliai --profile tts-loose --labeller whisper-hf:openai/whisper-large-v3-turbo
dp export     --work var/dataprep/bengaliai --profile tts-loose --dataset tts-loose --out var/import/bengaliai-tts-loose
```

`tts-strict` is selected and exported the same way. To relax thresholds for a trial run, repeat `select --override column.min=value` (or `.max`; `.lo`/`.hi` for percentile filters, `min_speaker_minutes=N`); the overrides are recorded in the selection report.

## 3. TTS stage B, the recorded voice (spec 5.3)

```bash
dp import voice --src <dir with metadata.csv and wavs/> --work var/dataprep/voice
dp check-voice --work var/dataprep/voice --labeller whisper-hf:openai/whisper-large-v3-turbo   # runs the score steps, writes check_voice.html/.json
dp select --work var/dataprep/voice --profile voice
dp export --work var/dataprep/voice --profile voice --dataset voice \
  --speaker <speaker name> --provenance provenance.json --test-count 40 --out var/import/voice
```

`--provenance` is a JSON file with the ingest form's fields (`source`, `license`, `consent`, `commercial_ok`, `notes`) and is required for any non-Bengali.AI work dir. `--speaker` gives every clip one speaker name instead of the clustered pseudo-speakers. `--test-count` holds out that many takes as `split=test` (round-robin over `category` when present); the rest keep an empty split for the platform's default ratios. Brouhaha must have run (`check-voice` does it) before the `voice` select.

## 4. CosyVoice round-trip gate (spec 5.4, 12)

Uses the `cosyvoice` extra, which replaces `labeller` in the venv; the Whisper STT it needs is part of that extra.

```bash
bash tools/dataprep/scripts/setup_cosyvoice.sh          # syncs the cosyvoice extra, clones CosyVoice, downloads the weights
uv run --project tools/dataprep --no-sync python -m dataprep roundtrip-cosyvoice run \
  --src var/import/voice --out var/dataprep/roundtrip/voice
# listen to report.html, save verdicts.json, then
uv run --project tools/dataprep --no-sync python -m dataprep roundtrip-cosyvoice finalize --out var/dataprep/roundtrip/voice
```

Afterwards `uv sync --project tools/dataprep --extra score --extra labeller` restores the other venv.

## Eval text check (spec 10.2)

Report domain eval sentences that already appear in training transcripts (compared after `bn_normalize(mode="asr")`). Exit code 1 if any match, 0 otherwise; `#` comments and blank lines in the sentences file are skipped.

```bash
dp check-eval-text --sentences eval_sentences.txt --against var/dataprep/bengaliai --against var/dataprep/voice
```

## Tests

The dataprep venv is Linux-only, so tests run in the backend venv:

```bash
PYTHONPATH=tools/dataprep uv run --project backend --with pytest --with pandas --with pyyaml --with pyarrow --with librosa --with bnunicodenormalizer --with scikit-learn python -m pytest tools/dataprep/tests -q -p no:cacheprovider --rootdir tools/dataprep
```

On the VM, `HF_TOKEN=... bash tools/dataprep/scripts/vm_smoke.sh` runs every step with real models on a 20-clip fixture, exports asr-v1 and tts-loose, ingests them with the backend's `ingest_dir`, and prints `SMOKE OK`.
