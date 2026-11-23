@echo off
rem Run the whole platform locally on Windows for development.
rem
rem   run-local.bat          Postgres + MLflow in Docker if Docker is running, else SQLite without MLflow
rem   run-local.bat sqlite   force SQLite, no Docker, no MLflow
rem   run-local.bat stop     close the API, worker and web windows and stop the Docker services
rem
rem API http://localhost:8000, web http://localhost:3000, MLflow http://localhost:5000 (Docker mode).
rem Data goes to .\var (git-ignored). Environment variables set here override the repo .env file.

setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%CD%"
set "LOCAL_COMPOSE=deploy\local\docker-compose.yml"

if /i "%~1"=="stop" goto :stop

where uv >nul 2>&1 || (echo [run-local] uv not found. Install it from https://docs.astral.sh/uv/ & exit /b 1)
where npm >nul 2>&1 || (echo [run-local] npm not found. Install Node.js 20+. & exit /b 1)

rem ---- Local settings (override .env, which holds the VM values) ----
set "DATA_ROOT=%ROOT%\var"
set "BACKENDS_ROOT=%ROOT%\backends"
set "RECIPES_DIR=%ROOT%\recipes"
set "WEB_ORIGIN=http://localhost:3000"
set "NEXT_PUBLIC_API_URL=http://localhost:8000"
set "DISK_WARN_GB=5"
set "DISK_MIN_GB=2"
set "CANCEL_GRACE_S=10"
set "PYTHONUTF8=1"
if not exist "%DATA_ROOT%" mkdir "%DATA_ROOT%"

set "MODE=sqlite"
if /i not "%~1"=="sqlite" (
  docker info >nul 2>&1 && set "MODE=docker"
)

if "%MODE%"=="docker" (
  echo [run-local] Starting Postgres and MLflow in Docker...
  docker compose -f "%LOCAL_COMPOSE%" up -d --build --wait
  if errorlevel 1 (echo [run-local] Docker services failed. Try: run-local.bat sqlite & exit /b 1)
  set "DATABASE_URL=postgresql+asyncpg://mlops:localdev@localhost:5432/mlops"
  set "MLFLOW_TRACKING_URI=http://localhost:5000"
) else (
  echo [run-local] Docker not running or sqlite requested: using SQLite, MLflow disabled.
  set "DATABASE_URL=sqlite+aiosqlite:///%ROOT:\=/%/var/mlops.db"
  set "MLFLOW_TRACKING_URI="
)

echo [run-local] Installing Python dependencies...
uv sync --frozen
if errorlevel 1 exit /b 1

echo [run-local] Migrating the database...
uv run alembic -c api/alembic.ini upgrade head
if errorlevel 1 (echo [run-local] Migration failed. & exit /b 1)

if not exist "web\node_modules" (
  echo [run-local] Installing web dependencies...
  pushd web
  call npm ci
  if errorlevel 1 (popd & exit /b 1)
  popd
)

echo [run-local] Starting API, worker and web in separate windows...
start "BanglaVoice API" cmd /k uv run uvicorn mlops_api.app:create_app --factory --host 127.0.0.1 --port 8000
start "BanglaVoice Worker" cmd /k uv run python -m mlops_worker
start "BanglaVoice Web" /d "%ROOT%\web" cmd /k npm run dev

echo [run-local] Waiting for the API...
set /a TRIES=0
:wait_api
curl -fsS http://127.0.0.1:8000/api/health >nul 2>&1 && goto :api_up
set /a TRIES+=1
if %TRIES% geq 60 (echo [run-local] API did not start in 120 s. Check the "BanglaVoice API" window. & exit /b 1)
rem ping as a sleep: timeout fails when input is redirected
ping -n 3 127.0.0.1 >nul
goto :wait_api

:api_up
echo.
echo [run-local] Running (%MODE% mode).
echo   Web     http://localhost:3000
echo   API     http://localhost:8000/api/health
if "%MODE%"=="docker" echo   MLflow  http://localhost:5000
echo   Data    %DATA_ROOT%
echo   Stop:   run-local.bat stop
start "" http://localhost:3000
exit /b 0

:stop
echo [run-local] Stopping API, worker and web...
taskkill /FI "WINDOWTITLE eq BanglaVoice API*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq BanglaVoice Worker*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq BanglaVoice Web*" /T /F >nul 2>&1
docker info >nul 2>&1 && docker compose -f "%LOCAL_COMPOSE%" stop
echo [run-local] Stopped. Data is kept in .\var and the Docker volumes.
exit /b 0
