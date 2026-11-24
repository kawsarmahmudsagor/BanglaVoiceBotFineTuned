from __future__ import annotations

import pandas as pd
import pytest
from app.core.text import bn_normalize

from dataprep.__main__ import main
from dataprep.steps.text import clean
from dataprep.workdir import StepFile, WorkDir

YA_DECOMPOSED = "য়"
YA_PRECOMPOSED = "য়"

TABLE = [
    ("পোয়াতি", "পোয়াতি"),
    (f"আ{YA_DECOMPOSED}", f"আ{YA_DECOMPOSED}"),
    (f"আ{YA_PRECOMPOSED}", f"আ{YA_DECOMPOSED}"),
    ("আমি‌ ভাত", "আমি ভাত"),
    ("আমি​ ভাত", "আমি ভাত"),
    ("আমি | ভাত", "আমি । ভাত"),
    ("আমি  ভাত   খাই ", "আমি ভাত খাই"),
    ("আমি Google ব্যবহার করি", "আমি Google ব্যবহার করি"),
    ("আমার ২০২৪ এবং 15 টাকা", "আমার ২০২৪ এবং 15 টাকা"),
    ("সে “হ্যাঁ” বলল", 'সে "হ্যাঁ" বলল'),
]
SAFE_INPUTS = [t for t, _ in TABLE]


@pytest.mark.parametrize(("raw", "expected"), TABLE)
def test_clean_table(raw, expected):
    assert clean(raw) == expected


def test_ya_variants_identical():
    assert clean(f"আ{YA_DECOMPOSED}") == clean(f"আ{YA_PRECOMPOSED}")


@pytest.mark.parametrize("raw", SAFE_INPUTS)
def test_clean_idempotent(raw):
    assert clean(clean(raw)) == clean(raw)


@pytest.mark.parametrize("mode", ["asr", "tts"])
@pytest.mark.parametrize("raw", SAFE_INPUTS)
def test_clean_roundtrip_property(raw, mode):
    assert bn_normalize(clean(raw), mode=mode) == bn_normalize(raw, mode=mode)


def _write_clips(wd: WorkDir, ids: list[str], texts: list[str]) -> None:
    clips = pd.DataFrame(
        {"id": ids, "src_path": ["x"] * len(ids), "text_raw": texts, "src_split": ["train"] * len(ids)}
    )
    StepFile.write(wd.clips_path, clips, step="import", version=1, params={})


def test_text_flags_empty_after_normalize(tmp_path):
    wd = WorkDir(tmp_path)
    _write_clips(wd, ["a", "b", "c"], ["।।", "আমি Google 5", "আমি ভাত"])
    assert main(["text", "--work", str(tmp_path), "--workers", "1"]) == 0
    df = StepFile.read(wd.step_path("text"), step="text", version=1).set_index("id")
    assert list(df.columns) == ["text_clean", "has_latin", "has_digits", "n_chars", "empty_asr", "error"]
    assert bool(df.loc["a", "empty_asr"]) and df.loc["a", "n_chars"] == 0
    assert df.loc["b", "has_latin"] and df.loc["b", "has_digits"]
    assert not df.loc["c", "empty_asr"] and df.loc["c", "n_chars"] == 6


def test_text_resume_and_force(tmp_path):
    wd = WorkDir(tmp_path)
    _write_clips(wd, ["a"], ["আমি"])
    main(["text", "--work", str(tmp_path), "--workers", "1"])
    _write_clips(wd, ["a", "b"], ["আমি", "ভাত"])
    main(["text", "--work", str(tmp_path), "--workers", "1"])
    assert sorted(pd.read_parquet(wd.step_path("text"))["id"]) == ["a", "b"]
    main(["text", "--work", str(tmp_path), "--workers", "1", "--force"])
    assert len(pd.read_parquet(wd.step_path("text"))) == 2
