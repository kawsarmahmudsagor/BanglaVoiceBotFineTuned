#!/usr/bin/env bash
# Register the default training dataset (bengaliai-200h) so the train form preselects it.
# Idempotent: does nothing if the dataset already exists or the zip is missing.
# Needs the API up. Reads DEFAULT_DATASET_ZIP (default ~/bengaliai_200h.zip) and DEFAULT_DATASET_NAME from .env.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/lib/env.sh

NAME="${DEFAULT_DATASET_NAME:-bengaliai-200h}"
ZIP="${DEFAULT_DATASET_ZIP:-$HOME/bengaliai_200h.zip}"
API="${API_URL:-http://localhost:8000}"
DEST="$DATA_ROOT/import/$NAME"

say() { echo "[seed] $*"; }

if curl -fsS "$API/api/datasets" | grep -q "\"name\":\"$NAME\""; then
  say "dataset '$NAME' already registered"; exit 0
fi
if [ ! -f "$ZIP" ] && [ ! -f "$DEST/metadata_train.csv" ]; then
  say "no $ZIP and no $DEST: skipping default dataset"; exit 0
fi
if [ ! -f "$DEST/metadata_train.csv" ]; then
  say "unzipping $ZIP to $DEST ..."
  mkdir -p "$DEST.tmp"; unzip -q -o "$ZIP" -d "$DEST.tmp"; rm -rf "$DEST"; mv "$DEST.tmp" "$DEST"
fi
say "ingesting '$NAME' (the worker validates and versions it; watch /datasets)"
curl -fsS -X POST "$API/api/datasets/ingest-local" -H 'Content-Type: application/json' -d "{
  \"path\": \"$NAME\", \"name\": \"$NAME\", \"task\": \"asr\",
  \"provenance\": {\"source\": \"Bengali.AI Speech Recognition (Kaggle bengaliai-speech), random 200 h subset\",
                   \"license\": \"Kaggle competition rules\", \"consent\": \"Kaggle competition data\",
                   \"commercial_ok\": false}}"
echo
