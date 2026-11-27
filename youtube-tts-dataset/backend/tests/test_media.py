import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from ytds.pipeline.download import DownloadError
from ytds.pipeline.media import ingest_local, local_video_id, needs_proxy, probe

needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg missing"
)


@pytest.mark.parametrize(
    "name,vcodec,expected",
    [
        ("a.mkv", "h264", True),
        ("a.mp4", "hevc", True),
        ("a.mp4", "h264", False),
        ("a.webm", "vp9", False),
        ("a.mp4", None, True),
    ],
)
def test_needs_proxy(name, vcodec, expected):
    assert needs_proxy(Path(name), vcodec) is expected


def test_local_video_id_stable_and_prefixed(tmp_path):
    a = tmp_path / "a.bin"
    a.write_bytes(b"x" * 3_000_000)
    b = tmp_path / "b.bin"
    b.write_bytes(b"x" * 3_000_000)
    assert local_video_id(a) == local_video_id(b)
    assert local_video_id(a).startswith("local_") and len(local_video_id(a)) == 6 + 11
    b.write_bytes(b"y")
    assert local_video_id(a) != local_video_id(b)


def _make(path: Path, *, audio: bool, video: bool = True) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if video:
        cmd += ["-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=2"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=2"]
    cmd += [str(path)]
    subprocess.run(cmd, check=True)


@needs_ffmpeg
def test_ingest_local_makes_wav_and_proxy(tmp_path):
    src = tmp_path / "My Talk.mkv"
    _make(src, audio=True)
    out = tmp_path / "vid" / "download"
    meta = ingest_local(src, out, lambda d, t: None)
    assert meta["title"] == "My Talk"
    assert 1.5 < meta["duration_s"] < 3
    assert meta["media"] == "media/preview.mp4"
    assert (out.parent / "media" / "preview.mp4").stat().st_size > 0
    with wave.open(str(out / "audio.wav")) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (48000, 1, 2)
    assert probe(out.parent / "media" / "preview.mp4")["vcodec"] == "h264"


@needs_ffmpeg
def test_ingest_local_no_proxy_when_playable(tmp_path):
    src = tmp_path / "ok.mp4"
    _make(src, audio=True)
    out = tmp_path / "vid" / "download"
    meta = ingest_local(src, out, lambda d, t: None)
    assert meta["media"] == "download/upload.mp4"
    assert (out / "upload.mp4").exists() and (out / "audio.wav").exists()


@needs_ffmpeg
def test_ingest_local_video_only_rejected(tmp_path):
    src = tmp_path / "silent.mkv"
    _make(src, audio=False)
    with pytest.raises(DownloadError, match="no audio track"):
        ingest_local(src, tmp_path / "vid" / "download", lambda d, t: None)
