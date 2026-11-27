import pytest

from ytds.config import Thresholds
from ytds.pipeline.select import ab_cer, pick, score, select_clip

TH = Thresholds()
GOOD = "আমার সোনার বাংলা আমি তোমায় ভালোবাসি"  # 30 chars: 3.0 s -> 10 cps
OTHER = "তুমি কোথায় যাচ্ছ আজকে সকালে"


def test_broken_scores_zero():
    assert score("", 3.0, TH) == 0.0


def test_clean_text_scores_one():
    assert score(GOOD, 3.0, TH) == pytest.approx(1.0)


def test_repeat_penalty():
    text = "না " * 4 + "আমি আজ বাজারে যাব"
    th = Thresholds(max_word_repeat=99)
    assert score(text, 2.0, th) == pytest.approx(1.0 - 0.1 * 2 - min(0.3, 0.02 * 0))  # cps 10, run 4


def test_rate_penalty_capped():
    th = Thresholds(max_chars_per_s=100)
    n = len(GOOD.replace(" ", ""))
    assert score(GOOD, 1.0, th) == pytest.approx(1.0 - 0.02 * (n - 20))  # below the cap
    assert score(GOOD, 0.5, th) == pytest.approx(1.0 - 0.3)              # capped


def test_b_penalties():
    assert score(GOOD, 3.0, TH, {"max_compression": 2.5, "retry_failed": True}) == pytest.approx(0.5)
    assert score(GOOD, 3.0, TH, {"max_compression": 2.0, "retry_failed": False}) == pytest.approx(1.0)


def test_tie_goes_to_b():
    assert pick(0.90, 0.87, TH) == "b"
    assert pick(0.95, 0.80, TH) == "a"


def test_ab_cer():
    assert ab_cer("", " । ") == 0.0
    assert ab_cer(GOOD, "") == 1.0
    assert ab_cer(GOOD, GOOD + "।") == 0.0


def _rows(a, b):
    return [{"text_a": a, "text_b": b, "b_compression": 1.5, "b_retry_failed": False, "broken": 0}]


def test_select_clip_disagree_flag():
    r = select_clip(_rows(GOOD, OTHER), 3.0, TH)
    assert r["flag"] == "disagree" and r["status"] == "flagged" and r["drop_reason"] is None
    assert r["cer"] > 0.25


def test_select_clip_both_broken_dropped():
    r = select_clip(_rows("", ""), 3.0, TH)
    assert r["status"] == "dropped" and r["drop_reason"] == "bad_stt" and r["flag"] is None


def test_select_clip_agree_auto():
    r = select_clip(_rows(GOOD, GOOD) + _rows(OTHER, OTHER), 6.0, TH)
    assert r["status"] == "auto" and r["text_pick"] == "b" and r["flag"] is None
    assert r["text_a"] == GOOD + " " + OTHER
