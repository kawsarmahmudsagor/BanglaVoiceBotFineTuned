"""Download stage: URL expansion and yt-dlp audio fetch (48 kHz mono 16-bit WAV)."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Callable

import yt_dlp


VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # ids name files and folders on disk


VIDEO_FORMAT = (
    "bestvideo[height<=720][vcodec^=avc1]+bestaudio[acodec^=mp4a]"
    "/best[height<=720][ext=mp4]/best[height<=720]"
)


class DownloadError(RuntimeError):
    """Wraps a yt-dlp / ffmpeg failure; message keeps the last error line."""


def _default_extract(url: str, download: bool = False) -> dict:
    opts = {"extract_flat": "in_playlist", "quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=download)


def expand_urls(text: str, extract=None) -> list[tuple[str, str]]:
    extract = extract or _default_extract
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for line in text.splitlines():
        url = line.strip()
        if not url:
            continue
        info = extract(url, download=False)
        if info.get("entries") is not None:
            items = [
                (e["id"], e.get("url") or f"https://www.youtube.com/watch?v={e['id']}")
                for e in info["entries"]
                if e and e.get("id")
            ]
        else:
            items = [(info["id"], url)]
        bad = [vid for vid, _ in items if not VIDEO_ID.match(str(vid))]
        if bad:
            raise ValueError(f"invalid video id {bad[0]!r} from {url}")
        for vid, vurl in items:
            if vid not in seen:
                seen.add(vid)
                out.append((vid, vurl))
    return out


def _last_error_line(exc: Exception) -> str:
    lines = [l for l in str(exc).splitlines() if l.strip()]
    return lines[-1] if lines else repr(exc)


def _locate_video(info: dict, out_dir: Path) -> Path:
    for item in info.get("requested_downloads") or []:
        fp = item.get("filepath")
        if fp and Path(fp).exists():
            return Path(fp)
    found = sorted(out_dir.glob("video.*"))
    if not found:
        raise DownloadError("downloaded video file not found")
    return found[0]


def download(
    video_id: str,
    url: str,
    out_dir: Path,
    on_progress: Callable[[int, int], None],
    cookies: Path | None = None,
    ydl_factory=None,
    *,
    keep_video: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ydl_factory = ydl_factory or yt_dlp.YoutubeDL

    def hook(d: dict) -> None:
        if d.get("status") != "downloading":
            return
        total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
        on_progress(int(d.get("downloaded_bytes") or 0), int(total))

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "raw.%(ext)s"),
        "noplaylist": True,
        "retries": 5,
        "fragment_retries": 5,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
    }
    if keep_video:
        opts["format"] = VIDEO_FORMAT
        opts["outtmpl"] = str(out_dir / "video.%(ext)s")
        opts["merge_output_format"] = "mp4"
        del opts["postprocessors"]
    if cookies:
        opts["cookiefile"] = str(cookies)

    source = out_dir / "raw.wav"
    media_rel = None
    final = out_dir / "audio.wav"
    try:
        with ydl_factory(opts) as ydl:
            info = ydl.extract_info(url, download=True)
        if keep_video:
            source = _locate_video(info, out_dir)
            if source.suffix.lower() != ".mp4":
                mp4 = out_dir / "video.mp4"
                remux = subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
                     "-c", "copy", str(mp4)],
                    capture_output=True, encoding="utf-8", errors="replace",
                )
                if remux.returncode == 0:
                    source.unlink(missing_ok=True)
                    source = mp4
                else:
                    mp4.unlink(missing_ok=True)
            media_rel = f"download/{source.name}"
        proc = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
             "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(final)],
            capture_output=True, encoding="utf-8", errors="replace",
        )
        if proc.returncode != 0:
            err = proc.stderr.strip()
            raise DownloadError(
                _last_error_line(RuntimeError(err)) if err
                else f"ffmpeg exited with code {proc.returncode}"
            )
    except yt_dlp.utils.DownloadError as exc:
        raise DownloadError(_last_error_line(exc)) from exc
    except FileNotFoundError as exc:
        raise DownloadError(f"ffmpeg not found: {exc}") from exc
    finally:
        for p in out_dir.glob("raw.*"):
            p.unlink(missing_ok=True)

    meta = {
        "title": info.get("title"),
        "channel": info.get("channel") or info.get("uploader"),
        "upload_date": info.get("upload_date"),
        "duration_s": info.get("duration"),
        "license": info.get("license"),
    }
    if keep_video:
        meta["media"] = media_rel
    return meta
