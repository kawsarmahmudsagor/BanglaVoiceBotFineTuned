#!/usr/bin/env bash
# Set up what `python -m dataprep roundtrip-cosyvoice` needs, on the Linux VM:
#   1. the dataprep venv with the `cosyvoice` extra,
#   2. the upstream CosyVoice source (not a pip package) with its Matcha-TTS submodule,
#   3. the Fun-CosyVoice3-0.5B-2512 weights.
# Re-running is safe: existing checkouts and downloads are reused.
#
# Env overrides:
#   COSYVOICE_REF   git ref to check out (default: main; the resolved commit is recorded in summary.json)
#   COSYVOICE_ROOT  checkout dir      (default: var/vendor/CosyVoice)
#   MODEL_DIR       weights dir       (default: var/models/Fun-CosyVoice3-0.5B)
set -euo pipefail

cd "$(dirname "$0")/../../.."   # repo root
COSYVOICE_REF=${COSYVOICE_REF:-main}
COSYVOICE_ROOT=${COSYVOICE_ROOT:-var/vendor/CosyVoice}
MODEL_DIR=${MODEL_DIR:-var/models/Fun-CosyVoice3-0.5B}
HF_REPO=FunAudioLLM/Fun-CosyVoice3-0.5B-2512

echo "== uv sync (tools/dataprep, extra cosyvoice)"
uv sync --project tools/dataprep --extra cosyvoice

echo "== CosyVoice source -> ${COSYVOICE_ROOT} @ ${COSYVOICE_REF}"
if [ ! -d "${COSYVOICE_ROOT}/.git" ]; then
  mkdir -p "$(dirname "${COSYVOICE_ROOT}")"
  git clone https://github.com/FunAudioLLM/CosyVoice.git "${COSYVOICE_ROOT}"
fi
git -C "${COSYVOICE_ROOT}" fetch --quiet origin
git -C "${COSYVOICE_ROOT}" checkout --quiet "${COSYVOICE_REF}"
git -C "${COSYVOICE_ROOT}" submodule update --init --recursive
echo "   commit: $(git -C "${COSYVOICE_ROOT}" rev-parse HEAD)"

echo "== weights ${HF_REPO} -> ${MODEL_DIR}"
uv run --project tools/dataprep --no-sync python - "${HF_REPO}" "${MODEL_DIR}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
PY

test -f "${MODEL_DIR}/cosyvoice3.yaml" || { echo "ERROR: ${MODEL_DIR}/cosyvoice3.yaml missing after download" >&2; exit 1; }
echo "== done. Run, for example:"
echo "   uv run --project tools/dataprep --no-sync python -m dataprep roundtrip-cosyvoice run --src <dataset dir> --out var/dataprep/roundtrip/<name>"
