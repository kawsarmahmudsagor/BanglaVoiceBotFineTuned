"""Local media helpers: probe, preview proxy and ingest of user-uploaded files."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from ytds.pipeline.download import DownloadError, _last_error_line

_PLAYABLE_SUFFIXES = {".mp4", ".webm"}
_PLAYABLE_CODECS = {"h264", "vp8", "vp9", "av1"}


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise DownloadError(f"{cmd[0]} not found: {exc}") from exc


def probe(path: Path) -> dict:
    proc = _run(["ffprobe", "-v", "error", "-show_entries",
                 "stream=codec_type,codec_name:format=duration", "-of", "json", str(path)])
    if proc.returncode != 0:
        err = proc.stderr.strip()
        raise DownloadError(_last_error_line(RuntimeError(err)) if err else "ffprobe failed")
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams") or []
    vcodec = next((s.get("codec_name") for s in streams if s.get("codec_type") == "video"), None)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    try:
        duration = float((data.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        duration = 0.0
    return {"duration_s": duration, "has_audio": has_audio, "vcodec": vcodec}


def needs_proxy(path: Path, vcodec: str | None) -> bool:
    return not (Path(path).suffix.lower() in _PLAYABLE_SUFFIXES and vcodec in _PLAYABLE_CODECS)


def make_proxy(src: Path, out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                 "-vf", r"scale=-2:min(720\,ih)", "-c:v", "libx264", "-preset", "veryfast",
                 "-crf", "26", "-c:a", "aac", "-b:a", "128k", str(out)])
    if proc.returncode != 0:
        err = proc.stderr.strip()
        raise DownloadError(_last_error_line(RuntimeError(err)) if err else "proxy failed")


def local_video_id(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return "local_" + h.hexdigest()[:11]


def extract_audio(src: Path, dest: Path) -> None:
    proc = _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn",
                 "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(dest)])
    if proc.returncode != 0:
        err = proc.stderr.strip()
        raise DownloadError(
            _last_error_line(RuntimeError(err)) if err
            else f"ffmpeg exited with code {proc.returncode}"
        )


def ingest_local(src: Path, out_dir: Path, on_progress: Callable[[int, int], None]) -> dict:
    src, out_dir = Path(src), Path(out_dir)
    info = probe(src)
    if not info["has_audio"]:
        raise DownloadError("no audio track")
    out_dir.mkdir(parents=True, exist_ok=True)
    size = src.stat().st_size
    on_progress(0, size)
    extract_audio(src, out_dir / "audio.wav")
    media: str | None = None
    if info["vcodec"] is not None:
        if needs_proxy(src, info["vcodec"]):
            proxy = out_dir.parent / "media" / "preview.mp4"
            try:
                make_proxy(src, proxy)
                media = "media/preview.mp4"
            except DownloadError:
                proxy.unlink(missing_ok=True)
        else:
            dest = out_dir / f"upload{src.suffix.lower()}"
            if src.resolve() != dest.resolve():
                shutil.copyfile(src, dest)
            media = f"download/{dest.name}"
    on_progress(size, size)
    return {"title": src.stem, "duration_s": info["duration_s"], "media": media}
