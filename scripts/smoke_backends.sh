#!/usr/bin/env bash
# End-to-end smoke test of the training backends through the API: ingest a tiny dataset, train 20 steps, evaluate,
# export, then warm-start a second run from the first. Run it on the VM with the stack and worker up (scripts/dev.sh).
#
#   bash scripts/smoke_backends.sh [backend...]     default: unsloth hf nemo f5 cosyvoice
#
# Env: API (default http://localhost:8000/api), SKIP_BACKENDS (space-separated, for a backend skipped at its VM
# gate), RUN_TIMEOUT (seconds per run, default 3600), INGEST_TIMEOUT (default 600).
# Needs curl and python3 (or python); the fixture builder needs numpy + soundfile (see scripts/smoke/).
set -euo pipefail

cd "$(dirname "$0")/.."
API="${API:-http://localhost:8000/api}"
RUN_TIMEOUT="${RUN_TIMEOUT:-3600}"
INGEST_TIMEOUT="${INGEST_TIMEOUT:-600}"
SKIP_BACKENDS="${SKIP_BACKENDS:-}"
STAMP="$(date +%s)"

PY=python3
command -v "$PY" >/dev/null 2>&1 || PY=python
command -v "$PY" >/dev/null 2>&1 || { echo "python3 or python is required" >&2; exit 2; }

if [ "$#" -gt 0 ]; then BACKENDS=("$@"); else BACKENDS=(unsloth hf nemo f5 cosyvoice); fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
FAILED=0

# jget '<python expr over d>' reads JSON on stdin, prints the expression's value (empty for None).
jget() {
  "$PY" -c 'import json,sys
d = json.load(sys.stdin)
v = eval(sys.argv[1])
print("" if v is None else (json.dumps(v) if isinstance(v, (dict, list, bool)) else v))' "$1"
}

# The first model of each backend, for the task its smoke run uses (the first key of MODELS in
# backend/workers/backends/<name>/mlops_backend/models.py). Fields: model|task|method|license|commercial_ok|export|extra hparams.
spec_for() {
  case "$1" in
    unsloth)   echo 'Qwen/Qwen3-4B-Instruct-2507|llm-sft|qlora|apache-2.0|true|["merged-safetensors","gguf"]|{"max_seq_len":512,"lora_r":16,"lora_alpha":32}' ;;
    hf)        echo 'openai/whisper-large-v3-turbo|asr|full|mit|true|["ct2"]|{}' ;;
    nemo)      echo 'ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large|asr|full|mit|true|["nemo"]|{}' ;;
    f5)        echo 'ai4bharat/IndicF5|tts|full|mit|true|["f5-ckpt"]|{"batch_size_frames":3200}' ;;
    cosyvoice) echo 'FunAudioLLM/Fun-CosyVoice3-0.5B-2512|tts|full|apache-2.0|true|["cosyvoice-dir"]|{"components":["llm","flow"]}' ;;
    *) return 1 ;;
  esac
}

fixture_for() {  # task -> file path
  case "$1" in
    llm-sft) echo "scripts/smoke/fixtures/llm.jsonl" ;;
    asr) echo "$WORK/fixtures/asr.zip" ;;
    tts) echo "$WORK/fixtures/tts.zip" ;;
  esac
}

metric_for() {
  case "$1" in llm-sft) echo perplexity ;; asr) echo wer ;; tts) echo rtf ;; esac
}

declare -A DATASET_REF  # task -> name@version

build_fixtures() {
  mkdir -p "$WORK/fixtures"
  if command -v uv >/dev/null 2>&1; then
    (cd backend && uv run python ../scripts/smoke/make_audio_fixtures.py "$WORK/fixtures") >/dev/null
  else
    "$PY" scripts/smoke/make_audio_fixtures.py "$WORK/fixtures" >/dev/null
  fi
}

# tus upload of one file; prints the upload id.
upload() {
  local file="$1" size name loc
  size="$(wc -c < "$file" | tr -d ' ')"
  name="$(basename "$file")"
  loc="$(curl -sS -D - -o /dev/null -X POST "$API/uploads" -H 'Tus-Resumable: 1.0.0' \
    -H "Upload-Length: $size" \
    -H "Upload-Metadata: filename $(printf '%s' "$name" | base64 | tr -d '\n')" \
    | tr -d '\r' | awk 'tolower($1)=="location:" {print $2}')"
  [ -n "$loc" ] || return 1
  curl -sS -f -o /dev/null -X PATCH "$API/uploads/${loc##*/}" -H 'Tus-Resumable: 1.0.0' \
    -H 'Content-Type: application/offset+octet-stream' -H 'Upload-Offset: 0' --data-binary "@$file" || return 1
  echo "${loc##*/}"
}

# ingest <task>: sets DATASET_REF[task] to name@version.
ingest_task() {
  local task="$1" file uid name body iid resp state waited=0
  [ -z "${DATASET_REF[$task]:-}" ] || return 0
  file="$(fixture_for "$task")"
  uid="$(upload "$file")" || { echo "upload failed for $file"; return 1; }
  name="smoke-${task}-${STAMP}"
  body="{\"upload_id\":\"$uid\",\"name\":\"$name\",\"task\":\"$task\",\"provenance\":{\"source\":\"smoke\",\"license\":\"internal\",\"consent\":\"true\",\"commercial_ok\":false}}"
  resp="$(curl -sS -f -X POST "$API/datasets/ingest" -H 'Content-Type: application/json' -d "$body")" \
    || { echo "ingest request rejected"; return 1; }
  iid="$(echo "$resp" | jget 'd["ingest_id"]')"
  while :; do
    resp="$(curl -sS -f "$API/ingests/$iid")"
    state="$(echo "$resp" | jget 'd["state"]')"
    case "$state" in
      done) break ;;
      failed) echo "ingest failed: $(echo "$resp" | jget 'd.get("error") or d.get("report")')"; return 1 ;;
    esac
    [ "$waited" -lt "$INGEST_TIMEOUT" ] || { echo "ingest timed out"; return 1; }
    sleep 2; waited=$((waited + 2))
  done
  DATASET_REF[$task]="$name@$(echo "$resp" | jget 'd["version"]["number"]')"
}

# recipe_json <recipe-name> <backend> <init-run-or-empty>
recipe_json() {
  local rname="$1" backend="$2" init="$3" spec
  spec="$(spec_for "$backend")"
  SPEC="$spec" RNAME="$rname" BACKEND="$backend" INIT="$init" TRAIN="${DATASET_REF[$(echo "$spec" | cut -d'|' -f2)]}" \
  "$PY" -c '
import json, os
model, task, method, lic, comm, export, extra = os.environ["SPEC"].split("|", 6)
hp = {"learning_rate": 1e-5, "epochs": 1, "max_steps": 20, "eval": {"max_samples": 4}}
hp.update(json.loads(extra))
spec = {"name": os.environ["RNAME"], "task": task, "backend": os.environ["BACKEND"], "base_model": model,
        "base_model_meta": {"license": lic, "commercial_ok": comm == "true"},
        "datasets": {"train": os.environ["TRAIN"]}, "method": {"type": method}, "hparams": hp,
        "export": json.loads(export)}
if os.environ["INIT"]:
    spec["init_from"] = {"run": os.environ["INIT"], "checkpoint": "final"}
print(json.dumps({"spec": spec}))'
}

# run_one <label> <backend> <recipe-name> <init-run-or-empty> <expect-parent: 0|1>; sets RUN_ID. Returns 1 on failure.
run_one() {
  local label="$1" backend="$2" rname="$3" init="$4" want_parent="$5"
  local task metric resp rid state waited=0 detail rdir parent n art
  task="$(spec_for "$backend" | cut -d'|' -f2)"
  metric="$(metric_for "$task")"
  curl -sS -o "$WORK/resp.json" -w '%{http_code}' -X POST "$API/recipes" -H 'Content-Type: application/json' \
    -d "$(recipe_json "$rname" "$backend" "$init")" > "$WORK/code"
  [ "$(cat "$WORK/code")" = 201 ] || { echo "FAIL $label: recipe rejected: $(cat "$WORK/resp.json")"; return 1; }
  resp="$(curl -sS -f -X POST "$API/runs" -H 'Content-Type: application/json' -d "{\"recipe\":\"$rname\"}")" \
    || { echo "FAIL $label: run not created"; return 1; }
  rid="$(echo "$resp" | jget 'd["id"]')"
  RUN_ID="$rid"
  while :; do
    detail="$(curl -sS -f "$API/runs/$rid")"
    state="$(echo "$detail" | jget 'd["state"]')"
    case "$state" in done|failed|cancelled) break ;; esac
    [ "$waited" -lt "$RUN_TIMEOUT" ] || { echo "FAIL $label: run $rid timed out after ${RUN_TIMEOUT}s (state $state)"; return 1; }
    sleep 10; waited=$((waited + 10))
  done
  [ "$state" = done ] || { echo "FAIL $label: run $rid ended $state: $(echo "$detail" | jget 'd.get("error")')"; return 1; }
  n="$(curl -sS -f "$API/runs/$rid/prepared" | jget 'len(d["refs"])')"
  [ "${n:-0}" -gt 0 ] || { echo "FAIL $label: run $rid has no prepared refs"; return 1; }
  echo "$detail" | jget "'$metric' in (d.get('eval') or {}).get('metrics', {})" | grep -qx true \
    || { echo "FAIL $label: run $rid eval metrics lack '$metric'"; return 1; }
  if [ "$want_parent" = 1 ]; then
    parent="$(echo "$detail" | jget 'd["parent_run_id"]')"
    [ "$parent" = "$init" ] || { echo "FAIL $label: run $rid parent_run_id is '$parent', expected '$init'"; return 1; }
  fi
  rdir="$(echo "$detail" | jget 'd["run_dir"]')"
  n="$(echo "$detail" | jget 'len(d["exports"])')"
  [ "${n:-0}" -gt 0 ] || { echo "FAIL $label: run $rid has no export artifacts"; return 1; }
  while IFS= read -r art; do
    [ -e "$rdir/export/$art" ] || { echo "FAIL $label: export artifact missing on disk: $rdir/export/$art"; return 1; }
  done < <(echo "$detail" | jget '"\n".join(a["path"] for a in d["exports"])')
}

smoke_backend() {
  local backend="$1" spec model task first
  spec="$(spec_for "$backend")" || { echo "FAIL $backend: unknown backend"; return 1; }
  model="$(echo "$spec" | cut -d'|' -f1)"
  task="$(echo "$spec" | cut -d'|' -f2)"
  ingest_task "$task" || { echo "FAIL $backend/$model: dataset ingest"; return 1; }
  run_one "$backend/$model" "$backend" "smoke-$backend-$STAMP" "" 0 || return 1
  echo "PASS $backend/$model"
  first="$RUN_ID"
  run_one "$backend/$model warm-start" "$backend" "smoke-$backend-warm-$STAMP" "$first" 1 || return 1
  echo "PASS $backend/$model warm-start"
}

build_fixtures
for b in "${BACKENDS[@]}"; do
  case " $SKIP_BACKENDS " in *" $b "*) echo "SKIP $b"; continue ;; esac
  smoke_backend "$b" || FAILED=1
done
[ "$FAILED" = 0 ] || { echo "smoke FAILED"; exit 1; }
echo "smoke OK"
