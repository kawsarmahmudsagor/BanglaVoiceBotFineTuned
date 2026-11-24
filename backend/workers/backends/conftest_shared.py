"""Pytest fixtures shared by every training backend's light tests.

Backends enable them from their tests/conftest.py with
`pytest_plugins = ["conftest_shared"]`; scripts/test_backend.sh puts this
directory on PYTHONPATH. Heavy imports (numpy, pyarrow) stay inside functions.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "heavy: needs GPU/model weights (VM only)")


def _sine_wavs(root: Path, n: int, sr: int, seconds: float = 0.5) -> list[str]:
    import numpy as np

    from app.core.backend_kit.audio import write_wav

    names = []
    t = np.arange(int(sr * seconds)) / sr
    for i in range(n):
        name = f"s{i}.wav"
        audio = (0.3 * np.sin(2 * np.pi * (220 + 55 * i) * t)).astype("float32")
        write_wav(root / "files" / name, audio, sr)
        names.append(name)
    return names


@pytest.fixture
def tiny_audio_ds(tmp_path):
    """Factory: tiny_audio_ds(sr=16000) -> (root, file_names) with sine-wave WAVs in root/files."""

    def make(sr: int = 16000, n: int = 4) -> tuple[Path, list[str]]:
        root = tmp_path / "audio_ds"
        return root, _sine_wavs(root, n, sr)

    return make


@pytest.fixture
def run_dir(tmp_path):
    """Factory: run_dir(task, base_model, hparams=None, **cfg) -> RunDir with a valid config.json.

    A 4-row dataset manifest (parquet) is written and referenced as datasets["train"] and
    datasets["eval"]. llm-sft rows carry messages_json; asr/tts rows carry file/text(/speaker).
    `texts` (4 strings) replaces the default "text {i}" transcripts; `seconds` sets the clip length.
    """

    def make(task: str, base_model: str, hparams: dict | None = None, *, texts: list[str] | None = None,
             seconds: float = 0.5, **overrides):
        import pyarrow as pa
        import pyarrow.parquet as pq

        from app.core.contract import DatasetRef, RunConfig, RunDir, write_config
        from app.core.splits import SplitConfig

        root = tmp_path / "dataset"
        root.mkdir(exist_ok=True)
        splits = ["train", "train", "val", "test"]
        rows = []
        if task == "llm-sft":
            for i, sp in enumerate(splits):
                msgs = [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
                rows.append({"sample_id": f"s{i}", "split": sp,
                             "messages_json": json.dumps(msgs), "tools_json": None})
        else:
            names = _sine_wavs(root, 4, 16000, seconds)
            for i, sp in enumerate(splits):
                row = {"sample_id": f"s{i}", "split": sp, "file": names[i],
                       "text": texts[i] if texts else f"text {i}", "duration_s": seconds, "sample_rate": 16000}
                if task == "tts":
                    row["speaker"] = "spk"
                rows.append(row)
        manifest = root / "manifest.parquet"
        pq.write_table(pa.Table.from_pylist(rows), manifest)
        ref = DatasetRef(name="tiny", version=1, hash="h" * 16, root=str(root),
                         manifest=str(manifest), splits=["train", "val", "test"],
                         split=SplitConfig())
        fields = dict(
            run_id="r1", attempt=1, resume=False, task=task, backend="test", base_model=base_model,
            method={"type": "lora"}, hparams=hparams or {}, export=[], seed=1,
            datasets={"train": ref, "eval": ref}, prepared_root=str(tmp_path / "prepared"),
        )
        fields.update(overrides)
        rd = RunDir(tmp_path / "run")
        rd.ensure()
        write_config(rd, RunConfig(**fields))
        return rd

    return make


@pytest.fixture
def tiny_tts_ds(tmp_path):
    """Factory: tiny_tts_ds(n=4) -> (root, rows). Rows carry sample_id, audio_path, text, speaker, duration_s;
    the sine-wave WAVs alternate 22.05 / 44.1 kHz; speakers alternate a / b."""

    def make(n: int = 4, seconds: float = 0.5) -> tuple[Path, list[dict]]:
        import numpy as np

        from app.core.backend_kit.audio import write_wav

        root = tmp_path / "tts_ds"
        rows = []
        for i in range(n):
            sr = 22050 if i % 2 == 0 else 44100
            t = np.arange(int(sr * seconds)) / sr
            path = root / "files" / f"t{i}.wav"
            write_wav(path, (0.3 * np.sin(2 * np.pi * (200 + 40 * i) * t)).astype("float32"), sr)
            rows.append({"sample_id": f"t{i}", "audio_path": str(path), "text": f"আমি  যাব {i}",
                         "speaker": "ab"[i % 2], "duration_s": seconds})
        return root, rows

    return make
