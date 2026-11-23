from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def _wav_bytes(x: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x.astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class Audio:
    @staticmethod
    def sine(seconds: float, sr: int = 16000, freq: float = 220.0, amp: float = 0.2) -> bytes:
        t = np.arange(int(seconds * sr)) / sr
        return _wav_bytes(amp * np.sin(2 * np.pi * freq * t), sr)

    @staticmethod
    def clipped(seconds: float, sr: int = 16000) -> bytes:
        t = np.arange(int(seconds * sr)) / sr
        return _wav_bytes(np.clip(3.0 * np.sin(2 * np.pi * 220 * t), -1, 1), sr)

    @staticmethod
    def silence(seconds: float, sr: int = 16000) -> bytes:
        return _wav_bytes(np.zeros(int(seconds * sr)), sr)


@pytest.fixture
def audio() -> type[Audio]:
    return Audio


@pytest.fixture
def make_zip(tmp_path):
    def _make(files: dict[str, bytes], metadata: str | bytes | None, top: str | None = None,
              name: str = "upload.zip", extra_members: dict[str, bytes] | None = None) -> Path:
        z = tmp_path / name
        prefix = f"{top}/" if top else ""
        with zipfile.ZipFile(z, "w") as zf:
            if metadata is not None:
                data = metadata.encode("utf-8") if isinstance(metadata, str) else metadata
                zf.writestr(prefix + "metadata.csv", data)
            for rel, content in files.items():
                zf.writestr(prefix + rel, content)
            for rel, content in (extra_members or {}).items():
                zf.writestr(rel, content)
        return z

    return _make


@pytest.fixture
def dir_with(tmp_path):
    def _make(files: dict[str, bytes], metadata: str | None, name: str = "extracted") -> Path:
        root = tmp_path / name
        root.mkdir()
        for rel, content in files.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content)
        if metadata is not None:
            (root / "metadata.csv").write_text(metadata, encoding="utf-8", newline="")
        return root

    return _make


@pytest.fixture
def jsonl(tmp_path):
    def _make(rows: list, name: str = "data.jsonl") -> Path:
        p = tmp_path / name
        p.write_text("\n".join(r if isinstance(r, str) else json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                     encoding="utf-8")
        return p

    return _make
