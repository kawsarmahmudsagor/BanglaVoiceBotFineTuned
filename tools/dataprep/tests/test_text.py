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
# Repairs bnunicodenormalizer makes that NFC alone does not: doubled vowel sign, split o-kar.
REPAIR_TABLE = [
    ("কাা", "কা"),
    ("কিি", "কি"),
    ("কাা।", "কা।"),
    ("“কাা”", '"কা"'),
    ("(কাা) abc 12", "(কা) abc 12"),
    ("কাা, কিি|", "কা, কি।"),
]
TABLE += REPAIR_TABLE
# ra + ZWJ + hasanta + ya (ra-phala) is a conjunct whose joiner must survive.
CONJUNCT_ZWJ = "র‍্য"
TABLE.append((f"আ{CONJUNCT_ZWJ}", f"আ{CONJUNCT_ZWJ}"))
# Joiners next to a hasanta (U+09CD) between Bengali letters are kept (spec 7.1 step 3), as bn_normalize keeps
# them: ra-phala (র ZWJ ্ য), ZWJ after the hasanta (র ্ ZWJ য), and the explicit-hasanta ক ্ ZWNJ ষ.
HASANTA_JOINERS = [
    "র‍্যাব",  # র‍্যাব
    "আমি র‍্যাব দেখেছি",
    "র্‍য",
    "ক্‌ষ",
    "লক্‌ষ্য ঠিক",
]
TABLE += [(t, t) for t in HASANTA_JOINERS]
SAFE_INPUTS = [t for t, _ in TABLE if t not in {r for r, _ in REPAIR_TABLE}]


@pytest.mark.parametrize(("raw", "expected"), TABLE)
def test_clean_table(raw, expected):
    assert clean(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), REPAIR_TABLE)
def test_repair_is_not_nfc_alone(raw, expected):
    import unicodedata

    assert unicodedata.normalize("NFC", raw) != clean(raw)  # NFC alone would not repair it
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


def test_error_rows_recorded_and_retry_only_errors(tmp_path, monkeypatch):
    from dataprep.steps import text as text_step

    wd = WorkDir(tmp_path)
    calls: list[str] = []
    real = text_step._row

    def counting(row):
        calls.append(row["id"])
        return real(row)

    monkeypatch.setattr(text_step, "_row", counting)
    args = ["text", "--work", str(tmp_path), "--workers", "1"]
    _write_clips(wd, ["ok", "bad", "gone"], ["আমি", None, "ভাত"])
    main(args)
    df = pd.read_parquet(wd.step_path("text")).set_index("id")
    assert df.loc["bad", "error"] and pd.isna(df.loc["ok", "error"])
    assert sorted(calls) == ["bad", "gone", "ok"]

    # default rerun: nothing recomputed, errored row stays errored; "gone" is dropped from clips
    calls.clear()
    _write_clips(wd, ["ok", "bad"], ["আমি", "ভাত"])
    main(args)
    assert calls == []
    df = pd.read_parquet(wd.step_path("text")).set_index("id")
    assert sorted(df.index) == ["bad", "ok"] and df.loc["bad", "error"]

    # --retry-errors recomputes only the errored row
    main([*args, "--retry-errors"])
    assert calls == ["bad"]
    df = pd.read_parquet(wd.step_path("text")).set_index("id")
    assert pd.isna(df.loc["bad", "error"]) and df.loc["bad", "text_clean"] == "ভাত"
    assert df.loc["ok", "text_clean"] == "আমি"
