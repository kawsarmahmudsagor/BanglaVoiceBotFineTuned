#!/usr/bin/env bash
# Run a backend's tests. Light (default): in the backend workspace venv, heavy tests deselected.
# --heavy: in the backend's own venv, all tests (VM only).
set -euo pipefail

cd "$(dirname "$0")/.."
. scripts/lib/env.sh

name="${1:-}"
[ -n "$name" ] || { echo "usage: $0 <backend> [--heavy]" >&2; exit 2; }
dir="backend/workers/backends/$name"
[ -d "$dir/tests" ] || { echo "no tests in $dir" >&2; exit 2; }

if [ "${2:-}" = "--heavy" ]; then
  exec uv run --project "$dir" pytest "$dir/tests"
fi
sep=":"
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) sep=";" ;; esac  # Windows Python splits PYTHONPATH on ";"
# server extra provides pyarrow/numpy/soundfile; PYTHONPATH makes conftest_shared importable (R2).
exec uv run --project backend --extra server --extra audio --with pytest \
  env PYTHONPATH="$dir${sep}backend/workers/backends" pytest "$dir/tests" -m "not heavy"
