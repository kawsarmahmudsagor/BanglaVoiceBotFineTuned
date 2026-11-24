#!/usr/bin/env bash
# Bring up the compose stack, wait for the API, then run the worker in the foreground.
set -euo pipefail

cd "$(dirname "$0")/.."
. scripts/lib/env.sh
# the api and mlflow containers run as this user, so files under DATA_ROOT stay writable by it
export UID
GID="$(id -g)"
export GID

if [ -z "${DATA_ROOT:-}" ]; then
  echo "DATA_ROOT is not set. Copy .env.example to .env and edit it." >&2
  exit 1
fi
mkdir -p "$DATA_ROOT" 2>/dev/null || true
if [ ! -d "$DATA_ROOT" ] || [ ! -w "$DATA_ROOT" ]; then
  echo "DATA_ROOT ($DATA_ROOT) is missing or not writable by $(id -un)." >&2
  echo "Create it and give it to this user, e.g.: sudo mkdir -p $DATA_ROOT && sudo chown $(id -u):$(id -g) $DATA_ROOT" >&2
  exit 1
fi

docker compose up -d --build

echo "Waiting for the API..."
for _ in $(seq 1 60); do
  if curl -fsS http://localhost:8000/api/health >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
if [ -z "${ready:-}" ]; then
  echo "API did not become healthy in 120s. Check: docker compose logs api" >&2
  exit 1
fi

echo
echo "From your PC, open the tunnel with:"
echo "  ssh -L 3000:localhost:3000 -L 8000:localhost:8000 -L 5000:localhost:5000 <user>@<vm-ip>"
echo "then browse http://localhost:3000"
echo

exec uv run --project backend --extra worker python -m workers
