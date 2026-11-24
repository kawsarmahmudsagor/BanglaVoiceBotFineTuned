from __future__ import annotations

import json
import os

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


def make_wd(tmp_path) -> WorkDir:
    wd = WorkDir(tmp_path / "bengaliai")
    n = len(SPLITS)
    ids = [f"c{i}" for i in range(n)]
    srcs = []
    t = np.arange(22050 * 2) / 22050
    for i, rid in enumerate(ids):
        p = tmp_path / f"{rid}.wav"
        sf.write(p, (0.3 * np.sin(2 * np.pi * (200 + 40 * i) * t)).astype("float32"), 22050)
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
        pd.DataFrame({"id": ids, "dur": 2.0, "src_sr": 22050, "clip_frac": 0.0,
                      "silence_share": 0.0, "lufs": -23.0, "error": None}),
        step="audio_stats", version=1, params={},
    )
    return wd


PROFILE = Profile(name="t", task="asr")


def prepared(tmp_path) -> WorkDir:
    wd = make_wd(tmp_path)
    select(wd, PROFILE)
    return wd


def test_export_passes_validate_audio(tmp_path):
    wd = prepared(tmp_path)
    out = export(wd, PROFILE, "bn-asr", workers=1)
    assert out == wd.export_dir("bn-asr")
    report, rows = validate_audio(out, "asr")
    assert not report.errors
    assert len(rows) == 5
    by_text = {r["text"]: r["split"] for r in rows}
    for i, s in enumerate(SPLITS):
        assert by_text[f"আমি ভাত খাই {i}"] == s
    data, sr = sf.read(out / "audio" / "c0.flac")
    assert sr == 16000 and data.ndim == 1
    cur = json.loads((out / "curation.json").read_text(encoding="utf-8"))
    assert set(cur) == {"source", "task", "profile", "step_versions", "report", "tool_commit", "provenance"}
    assert cur["task"] == "asr" and cur["profile"]["task"] == "asr"
    assert cur["source"] == "bengaliai"
    assert cur["provenance"]["commercial_ok"] is False
    assert {"source", "license", "consent", "notes"} <= set(cur["provenance"])
    assert cur["step_versions"]["text"] == 1
    assert not list(out.rglob("*.tmp-*"))


def test_export_replaces_not_rewrites(tmp_path):
    wd = prepared(tmp_path)
    out = export(wd, PROFILE, "d", workers=1)
    f = out / "audio" / "c1.flac"
    old = f.read_bytes()
    link = tmp_path / "elsewhere.flac"
    os.link(f, link)
    ino = f.stat().st_ino
    export(wd, PROFILE, "d", workers=1)
    assert link.read_bytes() == old
    assert f.stat().st_ino != ino


def test_export_removes_deselected(tmp_path):
    wd = prepared(tmp_path)
    out = export(wd, PROFILE, "d", workers=1)
    sel = pd.read_parquet(wd.selection_path("t"))
    StepFile.write(wd.selection_path("t"), sel[sel["id"] != "c2"], step="select", version=1, params={"profile": "t"})
    export(wd, PROFILE, "d", workers=1)
    assert not (out / "audio" / "c2.flac").exists()
    assert (out / "audio" / "c1.flac").exists()
    report, rows = validate_audio(out, "asr")
    assert not report.errors and len(rows) == 4


def test_export_refuses_empty_selection(tmp_path):
    wd = prepared(tmp_path)
    sel = pd.read_parquet(wd.selection_path("t"))
    StepFile.write(wd.selection_path("t"), sel.iloc[0:0], step="select", version=1, params={"profile": "t"})
    with pytest.raises(ExportError, match="empty"):
        export(wd, PROFILE, "d", workers=1)


def test_export_needs_selection(tmp_path):
    wd = make_wd(tmp_path)
    with pytest.raises(ExportError, match="select"):
        export(wd, PROFILE, "d", workers=1)


def test_export_ignores_foreign_parquet_in_work_dir(tmp_path):
    wd = prepared(tmp_path)
    pd.DataFrame({"x": [1, 2]}).to_parquet(wd.root / "notes.parquet")  # no dataprep metadata
    out = export(wd, PROFILE, "bn-asr", workers=1)
    cur = json.loads((out / "curation.json").read_text(encoding="utf-8"))
    assert cur["step_versions"]["text"] == 1 and "notes" not in json.dumps(cur["step_versions"])


def test_export_aborts_on_failed_transcode(tmp_path):
    wd = prepared(tmp_path)
    (tmp_path / "c3.wav").write_bytes(b"junk")
    with pytest.raises(ExportError, match="c3"):
        export(wd, PROFILE, "d", workers=1)
    assert not (wd.export_dir("d") / "metadata.csv").exists()
