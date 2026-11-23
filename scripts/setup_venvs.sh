#!/usr/bin/env bash
# Sync the root uv workspace, then one isolated venv per non-fake backend.
set -euo pipefail

cd "$(dirname "$0")/.."
. scripts/lib/env.sh

echo "== uv sync (backend workspace)"
uv sync --project backend --all-extras

status=0
for pyproject in backend/workers/backends/*/pyproject.toml; do
  [ -f "$pyproject" ] || continue
  dir=$(dirname "$pyproject")
  name=$(basename "$dir")
  [ "$name" = "fake" ] && continue

  # Backends must not be workspace members; they need their own venv and lockfile.
  if ! grep -Eq "\"workers/backends/${name}\"" <(sed -n '/^\[tool\.uv\.workspace\]/,/^\[/p' backend/pyproject.toml | grep -E '^exclude'); then
    echo "ERROR: workers/backends/${name} is not listed in [tool.uv.workspace] exclude in backend/pyproject.toml." >&2
    echo "       Add \"workers/backends/${name}\" to exclude, then re-run." >&2
    status=1
    continue
  fi

  echo "== uv sync --project ${dir}"
  uv sync --project "$dir"
  echo "   python: $(cd "$dir" && uv run --no-sync python -c 'import sys; print(sys.executable)')"
done

echo "== backend python: $(uv run --project backend --no-sync python -c 'import sys; print(sys.executable)')"
exit "$status"
