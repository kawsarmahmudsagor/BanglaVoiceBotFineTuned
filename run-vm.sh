#!/usr/bin/env bash
# Run the whole platform on the Ubuntu GPU VM: the Linux counterpart of run-local.bat.
#
#   ./run-vm.sh            check .env and the VM, sync venvs, start Docker services, start the worker in tmux
#   ./run-vm.sh --no-sync  same, but skip the uv venv sync (faster restart when dependencies did not change)
#   ./run-vm.sh status     show services, API health and worker state
#   ./run-vm.sh logs       attach to the worker (detach with Ctrl-b d; Ctrl-c there stops the worker)
#   ./run-vm.sh stop       stop the worker (graceful), then the Docker services; data and volumes are kept
#
# Postgres, MLflow, API and web run in Docker Compose (docker-compose.yml). The worker runs natively in a
# detached tmux session so training keeps going when SSH drops. Its output is also in $DATA_ROOT/logs/worker.log.
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
SESSION="mlops-worker"

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
  for pair in "postgres:${POSTGRES_PORT:-5432}" "mlflow:5000" "api:8000" "web:3000"; do
    name="${pair%%:*}"; port="${pair##*:}"
    grep -qx "$name" <<<"$running" && continue
    if ss -Hltn "sport = :$port" 2>/dev/null | grep -q .; then
      if [ "$name" = postgres ]; then
        die "port $port is already in use on this VM (see: sudo ss -ltnp 'sport = :$port'). Set POSTGRES_PORT to a free port in .env (e.g. 5433) and use the same port in DATABASE_URL"
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

  local ip
  ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  echo
  say "Running."
  echo "  From your PC:  ssh -L 3000:localhost:3000 -L 8000:localhost:8000 -L 5000:localhost:5000 $(id -un)@${ip:-<vm-ip>}"
  echo "  Then browse:   http://localhost:3000   (MLflow http://localhost:5000)"
  echo "  Worker:        ./run-vm.sh logs   (log file $DATA_ROOT/logs/worker.log)"
  echo "  Status / stop: ./run-vm.sh status | ./run-vm.sh stop"
}

case "${1:-}" in
  ""|--no-sync) cmd_start "${1:-}" ;;
  status) cmd_status ;;
  logs) cmd_logs ;;
  stop) cmd_stop ;;
  -h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) die "unknown command: $1 (use: ./run-vm.sh [--no-sync|status|logs|stop])" ;;
esac
