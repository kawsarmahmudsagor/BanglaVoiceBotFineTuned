from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from dataprep.__main__ import main
from dataprep.steps.audio_stats import audio_stats
from dataprep.workdir import StepFile, WorkDir


def _tone_then_silence(path, sr=22050):
    t = np.arange(2 * sr) / sr
    x = np.concatenate([0.5 * np.sin(2 * np.pi * 440 * t), np.zeros(sr)]).astype("float32")
    sf.write(path, x, sr)


def _write_clips(wd: WorkDir, paths: dict[str, object]) -> None:
    clips = pd.DataFrame(
        {
            "id": list(paths),
            "src_path": [str(p) for p in paths.values()],
            "text_raw": ["x"] * len(paths),
            "src_split": ["train"] * len(paths),
        }
    )
    StepFile.write(wd.clips_path, clips, step="import", version=1, params={})


def test_audio_stats_values(tmp_path):
    p = tmp_path / "a.wav"
    _tone_then_silence(p)
    s = audio_stats(p)
    assert abs(s["dur"] - 3.0) < 0.01
    assert s["src_sr"] == 22050
    assert s["clip_frac"] == 0
    assert abs(s["silence_share"] - 1 / 3) < 0.02
    assert s["lufs"] is not None and s["lufs"] < 0


def test_audio_stats_clipped(tmp_path):
    p = tmp_path / "sq.wav"
    sr = 16000
    x = np.where(np.sin(2 * np.pi * 100 * np.arange(sr) / sr) >= 0, 1.0, -1.0).astype("float32")
    sf.write(p, x, sr, subtype="FLOAT")
    assert audio_stats(p)["clip_frac"] > 0.9


def test_audio_stats_short_clip_has_no_lufs(tmp_path):
    p = tmp_path / "short.wav"
    sf.write(p, (0.3 * np.sin(np.arange(4000) / 10)).astype("float32"), 16000)  # 0.25 s
    assert audio_stats(p)["lufs"] is None


def test_audio_stats_undecodable(tmp_path):
    good = tmp_path / "a.wav"
    _tone_then_silence(good)
    bad = tmp_path / "b.mp3"
    bad.write_bytes(b"this is not audio, just text")
    wd = WorkDir(tmp_path / "work")
    _write_clips(wd, {"good": good, "bad": bad})
    assert main(["audio-stats", "--work", str(wd.root), "--workers", "1"]) == 0
    df = pd.read_parquet(wd.step_path("audio_stats")).set_index("id")
    assert df.loc["bad", "error"] and pd.isna(df.loc["good", "error"])
    assert abs(df.loc["good", "dur"] - 3.0) < 0.01
    assert list(df.columns) == ["dur", "src_sr", "clip_frac", "silence_share", "lufs", "error"]


def test_resume_skips_done(tmp_path, monkeypatch):
    from dataprep.steps import audio_stats as step

    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    _tone_then_silence(a)
    _tone_then_silence(b)
    wd = WorkDir(tmp_path / "work")
    _write_clips(wd, {"a": a, "b": b})
    calls: list[str] = []
    real = step._row

    def counting(row):
        calls.append(row["id"])
        return real(row)

    monkeypatch.setattr(step, "_row", counting)
    args = ["audio-stats", "--work", str(wd.root), "--workers", "1"]
    main(args)
    assert sorted(calls) == ["a", "b"]
    calls.clear()
    main(args)
    assert calls == []
    assert len(pd.read_parquet(wd.step_path("audio_stats"))) == 2


def test_checkpoint_size_default_and_explicit():
    from dataprep.steps._runner import checkpoint_size

    assert checkpoint_size(10, None) == 4096
    assert checkpoint_size(1_000_000, None) == 20_000  # <= 50 rewrites
    assert checkpoint_size(1_000_000, 64) == 64  # transcribe's explicit value wins


def _fake_stats(row: dict) -> dict:
    return {"id": row["id"], "dur": 1.0, "src_sr": 16000, "clip_frac": 0.0, "silence_share": 0.0, "lufs": -23.0}


def test_interrupt_keeps_checkpointed_rows_and_prints_progress(tmp_path, monkeypatch, capsys):
    """Checkpointing is on by default for every step: chunk = max(4096, ceil(n/50))."""
    import dataprep.steps.audio_stats as step

    n = 5000  # one 4096-row chunk, then an interrupt inside the second
    wd = WorkDir(tmp_path / "w")
    _write_clips(wd, {f"r{i:05d}": "x.wav" for i in range(n)})
    calls = {"n": 0}

    def flaky(row):
        calls["n"] += 1
        if calls["n"] > 4500:
            raise KeyboardInterrupt
        return _fake_stats(row)

    monkeypatch.setattr(step, "_row", flaky)
    with pytest.raises(KeyboardInterrupt):
        step.run(wd, workers=1)
    kept = pd.read_parquet(wd.step_path("audio_stats"))
    assert len(kept) == 4096 and kept["error"].isna().all()
    err = capsys.readouterr().err
    assert re.search(r"audio_stats: 4096/5000 rows .*rows/s.*ETA", err), err

    monkeypatch.setattr(step, "_row", _fake_stats)
    assert step.run(wd, workers=1) == n - 4096  # resume computes only the remainder
    assert len(pd.read_parquet(wd.step_path("audio_stats"))) == n
    assert "904/904 rows (100%)" in capsys.readouterr().err  # totals count this run's rows
