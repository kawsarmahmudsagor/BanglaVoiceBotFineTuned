#!/usr/bin/env bash
# Sync the root uv workspace, then one isolated venv per non-fake backend.
set -euo pipefail

cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi

echo "== uv sync (workspace root)"
uv sync

status=0
for pyproject in backends/*/pyproject.toml; do
  [ -f "$pyproject" ] || continue
  dir=$(dirname "$pyproject")
  name=$(basename "$dir")
  [ "$name" = "fake" ] && continue

  # Backends must not be workspace members; they need their own venv and lockfile.
  if ! grep -Eq "\"backends/${name}\"" <(sed -n '/^\[tool\.uv\.workspace\]/,/^\[/p' pyproject.toml | grep -E '^exclude'); then
    echo "ERROR: backends/${name} is not listed in [tool.uv.workspace] exclude in the root pyproject.toml." >&2
    echo "       Add \"backends/${name}\" to exclude, then re-run." >&2
    status=1
    continue
  fi

  echo "== uv sync --project ${dir}"
  uv sync --project "$dir"
  echo "   python: $(cd "$dir" && uv run --no-sync python -c 'import sys; print(sys.executable)')"
done

echo "== workspace python: $(uv run --no-sync python -c 'import sys; print(sys.executable)')"
exit "$status"
