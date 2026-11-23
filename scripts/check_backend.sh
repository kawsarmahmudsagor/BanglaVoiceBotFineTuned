#!/usr/bin/env bash
# Verify a backend's venv on the VM: torch + CUDA + key imports. Exits 1 on failure.
set -euo pipefail

cd "$(dirname "$0")/.."
. scripts/lib/env.sh

name="${1:-}"
[ -n "$name" ] || { echo "usage: $0 <backend>" >&2; exit 2; }
dir="backend/workers/backends/$name"
[ -f "$dir/pyproject.toml" ] || { echo "no backend at $dir" >&2; exit 2; }

uv sync --project "$dir"
uv run --project "$dir" python -m mlops_backend.check
