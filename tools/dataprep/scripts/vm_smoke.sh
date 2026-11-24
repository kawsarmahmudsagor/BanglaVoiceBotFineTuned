#!/usr/bin/env bash
# End-to-end smoke test of the dataprep CLI on the Linux VM, with real models, on a 20-clip fixture:
#   import -> text, audio-stats (--workers 2) -> dnsmos -> transcribe -> brouhaha, speakers (--only-profile
#   tts-loose, relaxed) -> select (4 profiles) -> export (asr-v1, tts-loose) -> check-eval-text (must exit 1)
#   -> ingest both exports in-process with the backend's ingest_dir (workers=2).
# Thresholds are relaxed with `select --override` so a 20-clip set keeps at least one row per profile.
# Prints "SMOKE OK" as the last line; the temp dir is removed on success and kept on failure.
#
# Needs: a GPU, HF_TOKEN (brouhaha / pyannote weights are gated), the Bengali.AI corpus.
# Env:
#   BENGALIAI_SRC  corpus dir with train.csv and train_mp3s/ (default: $HOME/.cache/kagglehub/competitions/bengaliai-speech)
#   HF_TOKEN       Hugging Face token (required)
#   STT_LABELLER   labeller spec (default: whisper-hf:openai/whisper-large-v3-turbo)
set -euo pipefail

cd "$(dirname "$0")/../../.."   # repo root
BENGALIAI_SRC=${BENGALIAI_SRC:-$HOME/.cache/kagglehub/competitions/bengaliai-speech}
STT_LABELLER=${STT_LABELLER:-whisper-hf:openai/whisper-large-v3-turbo}
N_CLIPS=20

: "${HF_TOKEN:?HF_TOKEN is required (brouhaha is gated on Hugging Face)}"
export HF_TOKEN
command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 || { echo "ERROR: no GPU visible (nvidia-smi failed)" >&2; exit 1; }
[ -f "${BENGALIAI_SRC}/train.csv" ] && [ -d "${BENGALIAI_SRC}/train_mp3s" ] \
  || { echo "ERROR: ${BENGALIAI_SRC} needs train.csv and train_mp3s/ (set BENGALIAI_SRC)" >&2; exit 1; }

SMOKE=var/dataprep/smoke-$(date +%Y%m%d-%H%M%S)
SRC=${SMOKE}/src
WORK=${SMOKE}/work
IMPORT=${SMOKE}/import     # export dirs; same filesystem as DEST so ingest hard-links
DEST=${SMOKE}/datasets
mkdir -p "${SRC}/train_mp3s" "${WORK}" "${IMPORT}" "${DEST}"
trap 'echo "SMOKE FAILED; work kept in ${SMOKE}" >&2' ERR

dp() { uv run --project tools/dataprep --no-sync python -m dataprep "$@"; }

echo "== fixture: first ${N_CLIPS} rows of ${BENGALIAI_SRC}/train.csv"
python3 - "${BENGALIAI_SRC}" "${SRC}" "${N_CLIPS}" <<'PY'
import csv, os, sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(os.path.join(src, "train.csv"), newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = [r for _, r in zip(range(n), reader)]
id_col = header.index("id")
with open(os.path.join(dst, "train.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)
for r in rows:
    os.symlink(os.path.abspath(os.path.join(src, "train_mp3s", r[id_col] + ".mp3")),
               os.path.join(dst, "train_mp3s", r[id_col] + ".mp3"))
print(f"{len(rows)} clips")
PY

echo "== uv sync (extra score)"
uv sync --project tools/dataprep --extra score

echo "== import, text, audio-stats (2 workers: the process pool path)"
dp import bengaliai --src "${SRC}" --work "${WORK}"
dp text --work "${WORK}" --workers 2
dp audio-stats --work "${WORK}" --workers 2

echo "== dnsmos (GPU)"
dp dnsmos --work "${WORK}"

echo "== uv sync (extras score + labeller), transcribe with ${STT_LABELLER}"
uv sync --project tools/dataprep --extra score --extra labeller
dp transcribe --work "${WORK}" --labeller "${STT_LABELLER}"

TTS_RELAX=(
  --override dur.min=0 --override dur.max=100 --override speech_span_s.min=0 --override clip_frac.max=1
  --override cps.lo=0 --override cps.hi=100 --override cer.max=100
  --override dnsmos_ovrl.min=0 --override dnsmos_bak.min=0 --override max_pause_s.max=1000
  --override min_speaker_minutes=0
)

echo "== brouhaha, speakers --only-profile tts-loose (GPU), as the README's TTS stage A runs them"
# --only-profile takes no --override, so a tts-loose copy with the same relaxed bounds (still named tts-loose)
# is written as YAML; on 20 clips the shipped thresholds could leave nothing to score.
dp_py() { uv run --project tools/dataprep --no-sync python "$@"; }
dp_py - "${SMOKE}/tts-loose.yaml" "${TTS_RELAX[@]}" <<'PY'
import sys

import yaml

from dataprep.profiles import load_profile
from dataprep.select import apply_overrides

out, args = sys.argv[1], [a for a in sys.argv[2:] if a != "--override"]
profile = apply_overrides(load_profile("tts-loose"), args)
with open(out, "w", encoding="utf-8") as f:
    yaml.safe_dump(profile.model_dump(mode="json"), f, allow_unicode=True)
print(f"relaxed tts-loose written to {out}")
PY
dp brouhaha --work "${WORK}" --only-profile "${SMOKE}/tts-loose.yaml" --labeller "${STT_LABELLER}"
dp speakers --work "${WORK}" --only-profile "${SMOKE}/tts-loose.yaml" --labeller "${STT_LABELLER}"

echo "== select, thresholds relaxed for a ${N_CLIPS}-clip set"
dp select --work "${WORK}" --profile asr-v1 \
  --override dur.min=0 --override cps.lo=0 --override cps.hi=100
dp select --work "${WORK}" --profile asr-v2 --labeller "${STT_LABELLER}" \
  --override dur.min=0 --override cps.lo=0 --override cps.hi=100 --override cer.max=100
dp select --work "${WORK}" --profile tts-loose --labeller "${STT_LABELLER}" "${TTS_RELAX[@]}"
dp select --work "${WORK}" --profile tts-strict --labeller "${STT_LABELLER}" "${TTS_RELAX[@]}" \
  --override snr_db.min=-1000 --override c50_db.min=-1000

echo "== export asr-v1 and tts-loose"
dp export --work "${WORK}" --profile asr-v1 --dataset asr-v1 --out "${IMPORT}/asr-v1"
dp export --work "${WORK}" --profile tts-loose --dataset tts-loose --out "${IMPORT}/tts-loose"

echo "== check-eval-text finds a training sentence"
python3 - "${SRC}/train.csv" "${SMOKE}/eval.txt" <<'PY'
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8") as f:
    row = next(csv.DictReader(f))
open(sys.argv[2], "w", encoding="utf-8").write("# eval sentences\n" + row["sentence"] + "\n")
PY
# Exit code 1 means "a sentence matched"; 0 (no match) or anything else (a crash, a traceback) fails the smoke.
rc=0
dp check-eval-text --sentences "${SMOKE}/eval.txt" --against "${WORK}" || rc=$?
if [ "${rc}" -ne 1 ]; then
  echo "ERROR: check-eval-text exited ${rc}; expected 1 (a training sentence must match)" >&2
  exit 1
fi

echo "== ingest_dir with workers=2 (in-process, backend venv: the spawn validation pool on Linux)"
uv run --project backend python - "${IMPORT}" "${DEST}" <<'PY'
import sys
from pathlib import Path

from app.datasets.ingest import ingest_dir


def main() -> None:
    import_root, dest_root = Path(sys.argv[1]), Path(sys.argv[2])
    for task, name in (("asr", "asr-v1"), ("tts", "tts-loose")):
        seen = []
        res = ingest_dir(import_root / name, task, dest_root / name, workers=2,
                         progress=lambda done, total: seen.append((done, total)))
        errors = [f"{i.code}: {i.message}" for i in res.report.errors]
        assert not errors, f"{name}: ingest errors: {errors}"
        assert res.n_samples > 0, f"{name}: nothing ingested"
        assert seen and seen[-1][0] == seen[-1][1], f"{name}: progress did not reach the total: {seen[-3:]}"
        print(f"{name}: {res.n_samples} samples, version {res.version_hash}")


if __name__ == "__main__":
    main()
PY

trap - ERR
rm -rf "${SMOKE}"
echo "SMOKE OK"
