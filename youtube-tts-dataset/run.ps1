<#
.SYNOPSIS
  Start the YouTube TTS dataset builder (Windows PowerShell 5.1).
.PARAMETER Rebuild
  Reinstall frontend packages and rebuild frontend/dist even if they look current. Without it, npm ci runs
  when package-lock.json is newer than the installed node_modules, and dist is rebuilt when any file in
  frontend/src, index.html, package.json or package-lock.json is newer than dist/index.html.
.PARAMETER Dev
  Start the backend and the Vite dev server (hot reload) in two windows instead.
#>
param(
    [switch]$Rebuild,
    [switch]$Dev
)

$ErrorActionPreference = 'Continue'  # native tools write progress to stderr; exit codes are checked instead
$root = $PSScriptRoot
$backend = Join-Path $root 'backend'
$frontend = Join-Path $root 'frontend'

function Stop-WithFix([string]$what, [string]$fix) {
    Write-Host "ERROR: $what" -ForegroundColor Red
    Write-Host "Fix:   $fix" -ForegroundColor Yellow
    exit 1
}

function Test-Tool([string]$name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

function Invoke-Checked([string]$exe, [string[]]$arguments, [string]$where) {
    Push-Location $where
    try {
        & $exe @arguments
        if ($LASTEXITCODE -ne 0) { Stop-WithFix "'$exe $($arguments -join ' ')' failed in $where" 'see the output above' }
    } finally {
        Pop-Location
    }
}

function Get-NewestWrite([string[]]$paths) {
    # Latest LastWriteTimeUtc over the given files and every file under the given folders.
    $newest = [datetime]::MinValue
    foreach ($p in $paths) {
        if (-not (Test-Path $p)) { continue }
        $item = Get-Item $p
        if ($item.PSIsContainer) {
            $files = @(Get-ChildItem $p -Recurse -File -ErrorAction SilentlyContinue)
        } else {
            $files = @($item)
        }
        foreach ($f in $files) {
            if ($f.LastWriteTimeUtc -gt $newest) { $newest = $f.LastWriteTimeUtc }
        }
    }
    return $newest
}

function Get-Port {
    # Same order as the backend: environment, youtube-tts-dataset/.env, repo-root .env.
    if ($env:YTDS_PORT) { return $env:YTDS_PORT }
    foreach ($file in @((Join-Path $root '.env'), (Join-Path (Split-Path $root -Parent) '.env'))) {
        if (Test-Path $file) {
            foreach ($line in Get-Content $file) {
                if ($line -match '^\s*YTDS_PORT\s*=\s*(\d+)\s*$') { return $Matches[1] }
            }
        }
    }
    return '8765'
}

# --- prerequisites
if (-not (Test-Tool 'ffmpeg')) { Stop-WithFix 'ffmpeg is not on PATH' 'winget install Gyan.FFmpeg  (then open a new terminal)' }
if (-not (Test-Tool 'nvidia-smi')) { Stop-WithFix 'nvidia-smi not found: no NVIDIA driver / CUDA GPU' 'install the latest NVIDIA driver from nvidia.com' }
& nvidia-smi -L | Out-Null
if ($LASTEXITCODE -ne 0) { Stop-WithFix 'nvidia-smi could not talk to the GPU' 'reinstall or update the NVIDIA driver' }
if (-not (Test-Tool 'uv')) { Stop-WithFix 'uv is not on PATH' 'winget install astral-sh.uv' }
if (-not (Test-Tool 'npm.cmd')) { Stop-WithFix 'Node.js / npm is not on PATH' 'winget install OpenJS.NodeJS.LTS  (Node 20+)' }
if (-not (Test-Tool 'deno')) {
    Write-Host 'WARNING: deno not found. yt-dlp needs a JavaScript runtime for some YouTube formats.' -ForegroundColor Yellow
    Write-Host '         Fix: winget install DenoLand.Deno' -ForegroundColor Yellow
}

# --- backend dependencies
Invoke-Checked 'uv' @('sync') $backend

# --- frontend
$dist = Join-Path $frontend 'dist'
$distIndex = Join-Path $dist 'index.html'
$lockFile = Join-Path $frontend 'package-lock.json'
$installedLock = Join-Path $frontend 'node_modules\.package-lock.json'  # written by npm ci / npm install

$needInstall = $Rebuild -or -not (Test-Path $installedLock)
if (-not $needInstall) {
    # package-lock.json changed (e.g. after git pull) since node_modules was installed
    $needInstall = (Get-NewestWrite @($lockFile)) -gt (Get-Item $installedLock).LastWriteTimeUtc
}
if ($needInstall) {
    Invoke-Checked 'npm.cmd' @('ci') $frontend
}
if (-not $Dev) {
    $needBuild = $Rebuild -or -not (Test-Path $distIndex)
    if (-not $needBuild) {
        # any frontend source newer than the built bundle means dist is stale
        $sources = @((Join-Path $frontend 'src'), (Join-Path $frontend 'index.html'),
                     (Join-Path $frontend 'package.json'), $lockFile)
        $needBuild = (Get-NewestWrite $sources) -gt (Get-Item $distIndex).LastWriteTimeUtc
    }
    if ($needBuild) {
        Invoke-Checked 'npm.cmd' @('run', 'build') $frontend
    }
}

$port = Get-Port
if ($Dev) {
    Start-Process powershell -WorkingDirectory $backend -ArgumentList @('-NoExit', '-Command', 'uv run python -m ytds')
    Start-Process powershell -WorkingDirectory $frontend -ArgumentList @('-NoExit', '-Command', '$env:YTDS_PORT=''' + $port + '''; npm.cmd run dev')
    Write-Host "Backend:  http://localhost:$port" -ForegroundColor Green
    Write-Host 'Frontend: http://localhost:5173  (Vite dev server, proxies /api and /ws to the backend)' -ForegroundColor Green
    exit 0
}

Write-Host "Open http://localhost:$port  (Ctrl+C to stop)" -ForegroundColor Green
Push-Location $backend
try {
    & uv run python -m ytds
} finally {
    Pop-Location
}
