#!/usr/bin/env bash
# Run the whole platform on the Ubuntu GPU VM: the Linux counterpart of run-local.bat.
#
#   ./run-vm.sh            check .env and the VM, sync venvs, start Docker services, start the worker in tmux
#   ./run-vm.sh --no-sync  same, but skip the uv venv sync (faster restart when dependencies did not change)
#   ./run-vm.sh status     show services, API health and worker state
#   ./run-vm.sh logs       attach to the worker (detach with Ctrl-b d; Ctrl-c there stops the worker)
#   ./run-vm.sh stop       stop the worker (graceful), then the Docker services; data and volumes are kept
#   ./run-vm.sh dev        no image builds: only Postgres in Docker; MLflow, API (--reload), web (next dev) and
#                          worker run natively in tmux. Code edits apply without rebuilding anything.
#   ./run-vm.sh restart-worker  restart only the worker (after edits to training or worker code)
#
# Postgres, MLflow, API and web run in Docker Compose (docker-compose.yml). The worker runs natively in a
# detached tmux session so training keeps going when SSH drops. Its output is also in $DATA_ROOT/logs/worker.log.
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
SESSION="mlops-worker"
DEV_SESSION="mlops-dev"

say() { echo "[run-vm] $*"; }
die() { echo "[run-vm] ERROR: $*" >&2; exit 1; }

load_env() {
  if [ ! -f .env ]; then
    cp .env.example .env
    die ".env was missing; created it from .env.example. Edit it (DATA_ROOT, REPO_ROOT, POSTGRES_PASSWORD, DATABASE_URL, HF_TOKEN), then re-run."
  fi
  . scripts/lib/env.sh
  export UID
  GID="$(id -g)"
  export GID
}

worker_state() {  # running | dead | absent
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then echo absent; return; fi
  if [ "$(tmux list-panes -t "$SESSION" -F '#{pane_dead}' | head -n1)" = "1" ]; then echo dead; else echo running; fi
}

api_healthy() { curl -fsS http://localhost:8000/api/health >/dev/null 2>&1; }

cmd_status() {
  load_env
  docker compose ps
  if api_healthy; then say "API: healthy"; else say "API: not answering on :8000"; fi
  say "worker: $(worker_state) (tmux session $SESSION; log $DATA_ROOT/logs/worker.log)"
  if tmux has-session -t "$DEV_SESSION" 2>/dev/null; then
    say "dev mode: running (tmux session $DEV_SESSION; windows mlflow, api, web; attach: tmux attach -t $DEV_SESSION)"
  fi
}

cmd_logs() {
  [ "$(worker_state)" = "absent" ] && die "worker is not running; start it with ./run-vm.sh"
  exec tmux attach -t "$SESSION"
}

cmd_stop() {
  load_env
  if [ "$(worker_state)" = "running" ]; then
    say "Stopping the worker (Ctrl-C, up to $(( ${CANCEL_GRACE_S:-60} + 30 )) s)..."
    tmux send-keys -t "$SESSION" C-c
    for _ in $(seq 1 $(( ${CANCEL_GRACE_S:-60} + 30 ))); do
      [ "$(worker_state)" = "running" ] || break
      sleep 1
    done
  fi
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  if tmux has-session -t "$DEV_SESSION" 2>/dev/null; then
    say "Stopping dev-mode MLflow, API and web..."
    tmux kill-session -t "$DEV_SESSION"
  fi
  say "Stopping Docker services..."
  docker compose down
  say "Stopped. Data stays in $DATA_ROOT and the pgdata volume."
}

check_env() {
  local missing=()
  for v in DATA_ROOT REPO_ROOT POSTGRES_PASSWORD DATABASE_URL; do
    [ -n "${!v:-}" ] || missing+=("$v")
  done
  [ "${#missing[@]}" -eq 0 ] || die "set ${missing[*]} in .env"
  [ "$POSTGRES_PASSWORD" != "change-me" ] || die "POSTGRES_PASSWORD is still change-me in .env"
  case "$DATABASE_URL" in
    *":${POSTGRES_PASSWORD}@"*) ;;
    *) die "DATABASE_URL in .env does not contain POSTGRES_PASSWORD; keep them in sync" ;;
  esac
  case "$DATABASE_URL" in
    *"@localhost:${POSTGRES_PORT:-5432}/"*|*"@127.0.0.1:${POSTGRES_PORT:-5432}/"*) ;;
    *) die "DATABASE_URL in .env must use localhost:${POSTGRES_PORT:-5432} (POSTGRES_PORT, default 5432)" ;;
  esac
  [ "$(cd "$REPO_ROOT" 2>/dev/null && pwd -P)" = "$(pwd -P)" ] \
    || die "REPO_ROOT in .env ($REPO_ROOT) must be this checkout ($ROOT)"
  [ -n "${HF_TOKEN:-}" ] || say "warning: HF_TOKEN is empty; gated models (IndicF5, Parler, IndicConformer, Gemma) will fail"
  for tool in docker uv tmux curl; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool is not installed"
  done
  mkdir -p "$DATA_ROOT" 2>/dev/null || true
  if [ ! -d "$DATA_ROOT" ] || [ ! -w "$DATA_ROOT" ]; then
    die "DATA_ROOT ($DATA_ROOT) is missing or not writable. Run: sudo mkdir -p $DATA_ROOT && sudo chown $(id -u):$(id -g) $DATA_ROOT"
  fi
  mkdir -p "$DATA_ROOT/logs"
}

check_ports() {
  # a port held by something other than this compose stack makes `docker compose up` fail mid-way
  local running port name
  running="$(docker compose ps --services --status running 2>/dev/null || true)"
  for pair in "postgres:${POSTGRES_PORT:-5432}" "mlflow:5000" "api:8000" "web:${WEB_PORT:-3000}"; do
    name="${pair%%:*}"; port="${pair##*:}"
    grep -qx "$name" <<<"$running" && continue
    if ss -Hltn "sport = :$port" 2>/dev/null | grep -q .; then
      if [ "$name" = postgres ]; then
        die "port $port is already in use on this VM (see: sudo ss -ltnp 'sport = :$port'). Set POSTGRES_PORT to a free port in .env (e.g. 5433) and use the same port in DATABASE_URL"
      fi
      if [ "$name" = web ]; then
        die "port $port is already in use on this VM (see: sudo ss -ltnp 'sport = :$port'). Set WEB_PORT to a free port in .env (e.g. 3001) and use the same port in WEB_ORIGIN"
      fi
      die "port $port ($name) is already in use on this VM (see: sudo ss -ltnp 'sport = :$port'); free it first"
    fi
  done
}

ensure_mlflow_db() {
  # deploy/postgres/init.sql creates it only on a fresh volume
  local user="${POSTGRES_USER:-mlops}"
  if ! docker compose exec -T postgres psql -U "$user" -d mlops -tAc \
      "SELECT 1 FROM pg_database WHERE datname='mlflow'" | grep -q 1; then
    say "Creating the mlflow database..."
    docker compose exec -T postgres psql -U "$user" -d mlops -c 'CREATE DATABASE mlflow;'
  fi
}

start_worker() {
  case "$(worker_state)" in
    running)
      say "Worker already running (tmux session $SESSION). To restart it: ./run-vm.sh stop, then ./run-vm.sh"
      return ;;
    dead)
      say "Previous worker exited; last lines:"
      tmux capture-pane -p -t "$SESSION" | grep -v '^$' | tail -n 15 || true
      tmux kill-session -t "$SESSION" ;;
  esac
  say "Starting the worker in tmux session $SESSION..."
  # the tmux server may predate .env changes, so the session loads .env itself
  tmux new-session -d -s "$SESSION" -c "$ROOT" \
    "bash -c '. scripts/lib/env.sh; set -o pipefail; uv run --project backend --extra worker python -m workers 2>&1 | tee -i -a \"\$DATA_ROOT/logs/worker.log\"; echo \"[worker exited with \$?]\"'"
  tmux set-option -t "$SESSION" remain-on-exit on >/dev/null
  sleep 5
  [ "$(worker_state)" = "running" ] || { tmux capture-pane -p -t "$SESSION" | tail -n 20; die "worker exited right after start"; }
}

dev_window() {  # name, command: one tmux window per service; output also goes to $DATA_ROOT/logs/<name>.log
  local name="$1" cmd="$2"
  local wrapped="bash -c '. scripts/lib/env.sh; set -o pipefail; $cmd 2>&1 | tee -i -a \"\$DATA_ROOT/logs/$name.log\"; echo \"[$name exited with \$?]\"'"
  if tmux has-session -t "$DEV_SESSION" 2>/dev/null; then
    tmux new-window -t "$DEV_SESSION" -n "$name" -c "$ROOT" "$wrapped"
  else
    tmux new-session -d -s "$DEV_SESSION" -n "$name" -c "$ROOT" "$wrapped"
    tmux set-option -t "$DEV_SESSION" remain-on-exit on >/dev/null
  fi
}

wait_http() {  # url, seconds, label
  local i
  for i in $(seq 1 "$2"); do curl -fsS -o /dev/null "$1" 2>/dev/null && return 0; sleep 1; done
  die "$3 did not answer on $1 in $2 s; see: tmux attach -t $DEV_SESSION (or $DATA_ROOT/logs/)"
}

cmd_dev() {
  load_env
  check_env
  local web_port="${WEB_PORT:-3000}" pg_port="${POSTGRES_PORT:-5432}" user="${POSTGRES_USER:-mlops}"

  if tmux has-session -t "$DEV_SESSION" 2>/dev/null; then
    say "Dev mode already running (tmux session $DEV_SESSION). Restart: ./run-vm.sh stop, then ./run-vm.sh dev"
    return
  fi
  # the Docker-mode containers hold the same ports
  say "Stopping Docker-mode MLflow, API and web if they run..."
  docker compose stop mlflow api web >/dev/null 2>&1 || true

  say "Starting Postgres (stock image, no build)..."
  docker compose up -d --wait postgres
  ensure_mlflow_db

  say "Syncing the backend venv and migrating the database..."
  uv sync --frozen --all-extras --project backend
  uv run --project backend --extra server alembic -c backend/alembic.ini upgrade head

  if [ ! -d frontend/node_modules ]; then
    say "Installing web dependencies (once)..."
    (cd frontend && npm ci)
  fi

  # MLflow server in its own cached uvx env (the backend venv has only the client, without flask).
  # The explicit +psycopg2 driver avoids SQLAlchemy 2.1 defaulting postgresql:// to psycopg 3.
  dev_window mlflow "uvx --from mlflow==3.18.0 --with psycopg2-binary mlflow server \
    --backend-store-uri postgresql+psycopg2://$user:\$POSTGRES_PASSWORD@localhost:$pg_port/mlflow \
    --artifacts-destination \$DATA_ROOT/mlflow --serve-artifacts --host 127.0.0.1 --port 5000"
  dev_window api "BACKENDS_ROOT=$ROOT/backend/workers/backends RECIPES_DIR=$ROOT/recipes CATALOG_DIR=$ROOT/catalog \
    MLFLOW_TRACKING_URI=http://localhost:5000 WEB_ORIGIN=http://localhost:$web_port \
    uv run --project backend --extra server uvicorn app.app:create_app --factory \
    --host 127.0.0.1 --port 8000 --reload --reload-dir backend/app"
  dev_window web "cd frontend && NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev -- -p $web_port"

  say "Waiting for MLflow, API and web..."
  wait_http http://localhost:5000 120 MLflow
  wait_http http://localhost:8000/api/health 120 API
  wait_http "http://localhost:$web_port" 180 web

  start_worker
  bash scripts/seed_default_dataset.sh || say "warning: default dataset seeding failed (see above)"

  echo
  say "Running in dev mode (no Docker builds)."
  echo "  Browse:        http://localhost:$web_port   (MLflow http://localhost:5000)"
  echo "  From your PC:  ssh -L $web_port:localhost:$web_port -L 8000:localhost:8000 -L 5000:localhost:5000 $(id -un)@<vm-ip>"
  echo "  Services:      tmux attach -t $DEV_SESSION   (windows mlflow, api, web; logs in $DATA_ROOT/logs/)"
  echo "  Worker:        ./run-vm.sh logs   (restart after training-code edits: ./run-vm.sh restart-worker)"
  echo "  API and web reload on code edits by themselves."
}

cmd_restart_worker() {
  load_env
  if [ "$(worker_state)" = "running" ]; then
    say "Stopping the worker (Ctrl-C, up to $(( ${CANCEL_GRACE_S:-60} + 30 )) s)..."
    tmux send-keys -t "$SESSION" C-c
    for _ in $(seq 1 $(( ${CANCEL_GRACE_S:-60} + 30 ))); do
      [ "$(worker_state)" = "running" ] || break
      sleep 1
    done
  fi
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  start_worker
}

cmd_start() {
  local sync=1
  [ "${1:-}" = "--no-sync" ] && sync=0
  load_env
  check_env

  say "Checking the VM..."
  bash scripts/check_vm.sh

  if [ "$sync" = 1 ]; then
    say "Syncing uv venvs (backend + every training backend)..."
    bash scripts/setup_venvs.sh
  fi

  check_ports
  say "Starting Postgres..."
  docker compose up -d --wait postgres
  ensure_mlflow_db

  say "Building and starting MLflow, API and web (first build takes a few minutes)..."
  docker compose up -d --build

  say "Waiting for the API (migrations run on start)..."
  for _ in $(seq 1 150); do api_healthy && break; sleep 2; done
  api_healthy || die "API did not become healthy in 300 s. Check: docker compose logs api"

  start_worker
  bash scripts/seed_default_dataset.sh || say "warning: default dataset seeding failed (see above)"

  local ip
  ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  echo
  say "Running."
  local web_port="${WEB_PORT:-3000}"
  echo "  From your PC:  ssh -L $web_port:localhost:$web_port -L 8000:localhost:8000 -L 5000:localhost:5000 $(id -un)@${ip:-<vm-ip>}"
  echo "  Then browse:   http://localhost:$web_port   (MLflow http://localhost:5000)"
  echo "  Worker:        ./run-vm.sh logs   (log file $DATA_ROOT/logs/worker.log)"
  echo "  Status / stop: ./run-vm.sh status | ./run-vm.sh stop"
}

case "${1:-}" in
  ""|--no-sync) cmd_start "${1:-}" ;;
  status) cmd_status ;;
  logs) cmd_logs ;;
  stop) cmd_stop ;;
  dev) cmd_dev ;;
  restart-worker) cmd_restart_worker ;;
  -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) die "unknown command: $1 (use: ./run-vm.sh [--no-sync|dev|status|logs|restart-worker|stop])" ;;
esac
