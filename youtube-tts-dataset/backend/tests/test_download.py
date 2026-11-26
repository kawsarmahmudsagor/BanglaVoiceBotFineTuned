import wave
from pathlib import Path

import pytest
import yt_dlp

from ytds.pipeline.download import DownloadError, download, expand_urls


def _write_wav(path: Path, rate: int = 22050, seconds: float = 0.2) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00\x00\x00" * int(rate * seconds))


def test_expand_dedupes_same_video_twice():
    def extract(url, download=False):
        return {"id": "abc", "title": "t"}

    out = expand_urls(
        "https://youtu.be/abc\nhttps://www.youtube.com/watch?v=abc\n", extract
    )
    assert out == [("abc", "https://youtu.be/abc")]


def test_expand_playlist_and_member_url_dedupe():
    def extract(url, download=False):
        if "playlist" in url:
            return {
                "entries": [
                    {"id": "a", "url": "https://www.youtube.com/watch?v=a"},
                    {"id": "b", "url": "https://www.youtube.com/watch?v=b"},
                ]
            }
        return {"id": "b"}

    text = "https://www.youtube.com/playlist?list=PL1\n\nhttps://youtu.be/b\nhttps://youtu.be/c\n"
    # c is not known to the fake extractor beyond its own id
    out = expand_urls(text, lambda u, download=False: {"id": "c"} if u.endswith("/c") else extract(u))
    assert [v for v, _ in out] == ["a", "b", "c"]


class FakeYDL:
    info = {
        "id": "vid1",
        "title": "A title",
        "channel": "A channel",
        "upload_date": "20240131",
        "duration": 12.5,
        "license": None,
    }
    error: Exception | None = None

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download=True):
        if self.error:
            raise self.error
        for hook in self.opts["progress_hooks"]:
            hook({"status": "downloading", "downloaded_bytes": 50, "total_bytes": 100})
            hook({"status": "downloading", "downloaded_bytes": 80, "total_bytes_estimate": 160})
        _write_wav(Path(self.opts["outtmpl"].replace("%(ext)s", "wav")))
        return dict(self.info)


def test_progress_callback_gets_bytes(tmp_path):
    calls = []
    download("vid1", "u", tmp_path, lambda d, t: calls.append((d, t)), ydl_factory=FakeYDL)
    assert calls == [(50, 100), (80, 160)]


def test_metadata_fields_mapped_and_wav_normalised(tmp_path):
    meta = download("vid1", "u", tmp_path, lambda d, t: None, ydl_factory=FakeYDL)
    assert meta == {
        "title": "A title",
        "channel": "A channel",
        "upload_date": "20240131",
        "duration_s": 12.5,
        "license": None,
    }
    with wave.open(str(tmp_path / "audio.wav")) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (48000, 1, 2)
    assert [p.name for p in tmp_path.iterdir()] == ["audio.wav"]


def test_error_line_kept(tmp_path):
    class Bad(FakeYDL):
        error = yt_dlp.utils.DownloadError("ERROR: [youtube] x: Private video")

    with pytest.raises(DownloadError, match="Private video"):
        download("x", "u", tmp_path, lambda d, t: None, ydl_factory=Bad)


@pytest.mark.network
def test_real_download_short_video(tmp_path):
    url = (Path(__file__).parent / "data" / "smoke_url.txt").read_text().strip()
    meta = download("smoke", url, tmp_path, lambda d, t: None)
    with wave.open(str(tmp_path / "audio.wav")) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (48000, 1, 2)
        assert w.getnframes() > 48000
    assert meta["title"]
    assert meta["duration_s"] > 0
