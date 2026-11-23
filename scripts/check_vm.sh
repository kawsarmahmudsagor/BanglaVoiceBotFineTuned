#!/usr/bin/env bash
# Preflight checks for the training VM. Prints OK, or every failure.
set -euo pipefail

cd "$(dirname "$0")/.."
. scripts/lib/env.sh

failures=()

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | paste -sd ';' -)"
else
  failures+=("nvidia-smi not working (NVIDIA driver missing or not loaded)")
fi

if docker compose version >/dev/null 2>&1; then
  echo "docker: $(docker compose version)"
  if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
    echo "docker nvidia runtime: present"
  else
    failures+=("docker has no nvidia runtime (install nvidia-container-toolkit and run: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker)")
  fi
else
  failures+=("docker compose not available")
fi

if command -v uv >/dev/null 2>&1; then
  echo "uv: $(uv --version)"
else
  failures+=("uv not installed")
fi

for tool in cmake gcc make git; do
  if command -v "$tool" >/dev/null 2>&1; then
    echo "$tool: present"
  else
    failures+=("$tool not on PATH (needed to build llama.cpp for GGUF export; install build-essential, cmake and git)")
  fi
done

min_gb="${DISK_MIN_GB:-10}"
if [ -z "${DATA_ROOT:-}" ]; then
  failures+=("DATA_ROOT is not set (copy .env.example to .env)")
else
  probe="$DATA_ROOT"
  while [ ! -e "$probe" ] && [ "$probe" != "/" ]; do probe=$(dirname "$probe"); done
  free_gb=$(df -BG --output=avail "$probe" | tail -n1 | tr -dc '0-9')
  echo "disk: ${free_gb} GB free on ${probe} (DATA_ROOT=${DATA_ROOT}, min ${min_gb} GB)"
  if [ "${free_gb:-0}" -lt "$min_gb" ]; then
    failures+=("only ${free_gb} GB free on DATA_ROOT, need at least ${min_gb} GB")
  fi
fi

if [ "${#failures[@]}" -eq 0 ]; then
  echo "OK"
else
  for f in "${failures[@]}"; do echo "FAIL: $f" >&2; done
  exit 1
fi
