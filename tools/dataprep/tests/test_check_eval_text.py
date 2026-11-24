from __future__ import annotations

import pandas as pd

from dataprep.__main__ import main
from dataprep.workdir import StepFile, WorkDir


def make_work(root, texts: dict[str, str]):
    wd = WorkDir(root)
    ids = list(texts)
    StepFile.write(
        wd.step_path("text"),
        pd.DataFrame({
            "id": ids, "text_clean": [texts[i] for i in ids], "has_latin": False, "has_digits": False,
            "n_chars": 5, "empty_asr": False, "error": None,
        }),
        step="text", version=2, params={},
    )
    return root


def write_sentences(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_match_is_reported_and_exit_1(tmp_path, capsys):
    w = make_work(tmp_path / "w", {"c1": "আমি ভাত খাই।", "c2": "অন্য বাক্য"})
    s = write_sentences(tmp_path / "s.txt", ["# comment", "", "তুমি কোথায় যাও", "আমি ভাত খাই"])
    code = main(["check-eval-text", "--sentences", str(s), "--against", str(w)])
    out = capsys.readouterr().out
    assert code == 1
    assert "4" in out and "আমি ভাত খাই" in out and "c1" in out and str(w) in out
    assert "c2" not in out


def test_no_match_exit_0_multiple_against(tmp_path, capsys):
    a = make_work(tmp_path / "a", {"c1": "আমি ভাত খাই"})
    b = make_work(tmp_path / "b", {"d1": "সে বই পড়ে"})
    s = write_sentences(tmp_path / "s.txt", ["তুমি কোথায় যাও"])
    assert main(["check-eval-text", "--sentences", str(s), "--against", str(a), "--against", str(b)]) == 0
    s2 = write_sentences(tmp_path / "s2.txt", ["সে বই পড়ে"])
    assert main(["check-eval-text", "--sentences", str(s2), "--against", str(a), "--against", str(b)]) == 1
    assert "d1" in capsys.readouterr().out


def test_sentence_with_empty_normal_form_is_skipped(tmp_path, capsys):
    # A clip whose transcript normalizes to nothing must not "match" a punctuation-only eval line.
    w = make_work(tmp_path / "w", {"c1": "।", "c2": "আমি ভাত খাই"})
    s = write_sentences(tmp_path / "s.txt", ["...", "।।", "তুমি কোথায় যাও"])
    assert main(["check-eval-text", "--sentences", str(s), "--against", str(w)]) == 0
    out = capsys.readouterr().out
    assert "c1" not in out and "skipped 2" in out
