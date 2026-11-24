from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from app.datasets.validate_audio import validate_audio

from dataprep.export import ExportError, export
from dataprep.profiles import Profile
from dataprep.select import select
from dataprep.workdir import StepFile, WorkDir

SPLITS = ["train", "train", "train", "val", None]
SRS = [44100, 44100, 16000, 24000, 44100]
PROFILE = Profile(name="t", task="tts")


def make_wd(tmp_path, *, brouhaha=True, speakers=True) -> WorkDir:
    wd = WorkDir(tmp_path / "bengaliai")
    n = len(SPLITS)
    ids = [f"c{i}" for i in range(n)]
    srcs = []
    for i, rid in enumerate(ids):
        sr = SRS[i]
        t = np.arange(sr * 3) / sr
        p = tmp_path / f"{rid}.wav"
        sf.write(p, (0.1 * np.sin(2 * np.pi * (200 + 40 * i) * t)).astype("float32"), sr)
        srcs.append(str(p))
    texts = [f"আমি ভাত খাই {i}" for i in range(n)]
    StepFile.write(
        wd.clips_path,
        pd.DataFrame({"id": ids, "src_path": srcs, "text_raw": texts, "src_split": SPLITS}),
        step="import", version=1, params={},
    )
    StepFile.write(
        wd.step_path("text"),
        pd.DataFrame({"id": ids, "text_clean": texts, "has_latin": False, "has_digits": False,
                      "n_chars": [len(x) for x in texts], "empty_asr": False, "error": None}),
        step="text", version=1, params={},
    )
    StepFile.write(
        wd.step_path("audio_stats"),
        pd.DataFrame({"id": ids, "dur": 3.0, "src_sr": SRS, "clip_frac": 0.0,
                      "silence_share": 0.0, "lufs": -23.0, "error": None}),
        step="audio_stats", version=1, params={},
    )
    if brouhaha:
        StepFile.write(
            wd.step_path("brouhaha"),
            pd.DataFrame({"id": ids, "snr_db": 20.0, "c50_db": 50.0, "max_pause_s": 0.1,
                          "speech_segments": [[[0.3, 1.0], [1.5, 2.5]]] * n, "error": None}),
            step="brouhaha", version=1, params={},
        )
    if speakers:
        StepFile.write(
            wd.step_path("speakers"),
            pd.DataFrame({"id": ids, "speaker": ["bai-1", "bai-1", "bai-2", "bai-2", "bai-1"], "error": None}),
            step="speakers", version=1, params={},
        )
    select(wd, PROFILE)
    return wd


def test_tts_export_passes_validate_audio(tmp_path):
    wd = make_wd(tmp_path)
    out = export(wd, PROFILE, "bn-tts", workers=1)
    report, rows = validate_audio(out, "tts")
    assert not report.errors
    assert len(rows) == 5
    assert {r["speaker"] for r in rows} == {"bai-1", "bai-2"}
    data, sr = sf.read(out / "wavs" / "c0.wav", always_2d=False)
    info = sf.info(out / "wavs" / "c0.wav")
    assert sr == 24000 and data.ndim == 1 and info.subtype == "PCM_16"
    assert abs(len(data) / sr - (2.5 + 0.15 - 0.15)) < 0.02  # trimmed to segments +- 150 ms
    assert not (out / "audio").exists()
    assert not list(out.rglob("*.tmp-*"))
    cur = json.loads((out / "curation.json").read_text(encoding="utf-8"))
    assert cur["band_limited"] == 1  # only c2 has a 16 kHz source
    assert cur["profile"]["task"] == "tts"
    header = (out / "metadata.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "file,text,speaker,split"


def test_tts_speaker_override_needs_no_speakers_file(tmp_path):
    wd = make_wd(tmp_path, speakers=False)
    out = export(wd, PROFILE, "voice", workers=1, speaker="voice-1")
    report, rows = validate_audio(out, "tts")
    assert not report.errors
    assert {r["speaker"] for r in rows} == {"voice-1"}


def test_tts_without_speakers_names_speaker_flag(tmp_path):
    wd = make_wd(tmp_path, speakers=False)
    with pytest.raises(ExportError, match="--speaker"):
        export(wd, PROFILE, "d", workers=1)
    assert not wd.export_dir("d").exists() or not (wd.export_dir("d") / "metadata.csv").exists()


def test_tts_needs_brouhaha_segments(tmp_path):
    wd = make_wd(tmp_path, brouhaha=False)
    with pytest.raises(ExportError, match="brouhaha"):
        export(wd, PROFILE, "d", workers=1)
    assert not wd.export_dir("d").exists()


def test_tts_lists_ids_missing_segments(tmp_path):
    wd = make_wd(tmp_path)
    b = pd.read_parquet(wd.step_path("brouhaha"))
    StepFile.write(wd.step_path("brouhaha"), b[b["id"] != "c3"], step="brouhaha", version=1, params={})
    with pytest.raises(ExportError, match="c3"):
        export(wd, PROFILE, "d", workers=1)
    assert not wd.export_dir("d").exists()


def test_tts_abort_on_failure_writes_no_metadata(tmp_path):
    wd = make_wd(tmp_path)
    (tmp_path / "c1.wav").write_bytes(b"junk")
    with pytest.raises(ExportError, match="c1"):
        export(wd, PROFILE, "d", workers=1)
    assert not (wd.export_dir("d") / "metadata.csv").exists()
    assert not list(wd.export_dir("d").rglob("*.tmp-*"))


def test_sweep_only_removes_layout_files(tmp_path):
    wd = make_wd(tmp_path)
    out = export(wd, PROFILE, "d", workers=1)
    (out / "wavs" / "stale.wav").write_bytes(b"x")
    (out / "wavs" / "notes.txt").write_text("keep")
    (out / "wavs" / "subdir").mkdir()
    (out / "wavs" / "c9.wav.tmp-1234").write_bytes(b"x")
    export(wd, PROFILE, "d", workers=1)
    assert not (out / "wavs" / "stale.wav").exists()
    assert not (out / "wavs" / "c9.wav.tmp-1234").exists()
    assert (out / "wavs" / "notes.txt").exists()
    assert (out / "wavs" / "subdir").is_dir()
    assert (out / "wavs" / "c0.wav").exists()


def test_metadata_replaced_before_sweep(tmp_path, monkeypatch):
    wd = make_wd(tmp_path)
    out = export(wd, PROFILE, "d", workers=1)
    sel = pd.read_parquet(wd.selection_path("t"))
    StepFile.write(wd.selection_path("t"), sel[sel["id"] != "c2"], step="select", version=1, params={"profile": "t"})

    seen = {}
    real_unlink = type(out).unlink

    def spy(self, *a, **k):
        seen.setdefault("meta", (out / "metadata.csv").read_text(encoding="utf-8"))
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(type(out), "unlink", spy)
    export(wd, PROFILE, "d", workers=1)
    assert "c2.wav" not in seen["meta"]
    assert not (out / "wavs" / "c2.wav").exists()


def test_tts_preflight_rejects_short_speech_span(tmp_path, monkeypatch):
    wd = make_wd(tmp_path)
    b = pd.read_parquet(wd.step_path("brouhaha"))
    b["speech_segments"] = [[[1.0, 1.05]] if i == "c1" else s for i, s in zip(b["id"], b["speech_segments"])]
    StepFile.write(wd.step_path("brouhaha"), b, step="brouhaha", version=1, params={})
    import librosa

    def no_decode(*a, **k):
        raise AssertionError("decoded before pre-flight failed")

    monkeypatch.setattr(librosa, "load", no_decode)
    with pytest.raises(ExportError, match="c1") as e:
        export(wd, PROFILE, "d", workers=1)
    assert "select" in str(e.value)
    assert not wd.export_dir("d").exists()
