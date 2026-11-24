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

  if [ "$name" = "cosyvoice" ]; then
    # CosyVoice is not a pip package: clone upstream at the pinned commit (line 1 of UPSTREAM_COMMIT).
    commit=$(head -n 1 "$dir/UPSTREAM_COMMIT" | tr -d '[:space:]')
    up="$dir/.upstream"
    if [ "$(git -C "$up" rev-parse HEAD 2>/dev/null || true)" != "$commit" ]; then
      echo "== cosyvoice upstream @ ${commit}"
      [ -d "$up/.git" ] || { rm -rf "$up"; git clone --recurse-submodules https://github.com/FunAudioLLM/CosyVoice "$up"; }
      git -C "$up" fetch --quiet origin "$commit" 2>/dev/null || git -C "$up" fetch --quiet origin
      git -C "$up" checkout --quiet "$commit"
      git -C "$up" submodule update --init --recursive
    fi
  fi

  echo "== uv sync --project ${dir}"
  uv sync --project "$dir"

  if [ "$name" = "cosyvoice" ]; then
    # Put upstream + Matcha-TTS on sys.path through a .pth file in the backend venv.
    site=$(cd "$dir" && uv run --no-sync python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    up_abs="$(cd "$dir/.upstream" && pwd)"
    printf '%s\n%s\n' "$up_abs" "$up_abs/third_party/Matcha-TTS" > "$site/cosyvoice_upstream.pth"
    echo "   wrote $site/cosyvoice_upstream.pth"
  fi
  echo "   python: $(cd "$dir" && uv run --no-sync python -c 'import sys; print(sys.executable)')"
done

echo "== backend python: $(uv run --project backend --no-sync python -c 'import sys; print(sys.executable)')"
exit "$status"
