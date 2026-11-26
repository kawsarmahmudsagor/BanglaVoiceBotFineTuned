from ytds.config import Thresholds
from ytds.pipeline.routing import bn_ratio, longest_repeat, route

TH = Thresholds()
SENTENCE = "আমার সোনার বাংলা আমি তোমায় ভালোবাসি"  # 30 Bangla characters without spaces


def test_normal_bangla_is_auto():
    assert route(SENTENCE, 3.0, TH) == "auto"


def test_empty_or_punctuation_only_is_fallback():
    assert route("", 3.0, TH) == "fallback"
    assert route(" । ", 3.0, TH) == "fallback"


def test_mostly_latin_is_fallback_but_loanwords_pass():
    assert route("this is all english text here", 3.0, TH) == "fallback"
    assert route("iPhone 15 এর দাম কত টাকা", 2.0, TH) == "auto"


def test_bn_ratio_ignores_digits_and_punctuation():
    assert bn_ratio("আমি ১৫, ১৬।") == 1.0
    assert bn_ratio("123 ...") == 0.0


def test_speaking_rate_bounds():
    assert route(SENTENCE, 15.0, TH) == "fallback"   # 2 chars/s: too few words for the audio
    assert route(SENTENCE, 0.5, TH) == "fallback"    # 60 chars/s: more text than anyone can say


def test_repeat_loop_is_fallback():
    assert longest_repeat("ছোট ছোট ছোট ছোট জীবন") == 4
    assert route("ছোট ছোট ছোট ছোট জীবনের ব্যাটল", 2.0, TH) == "auto"
    assert route("না " * 8 + "আমি", 2.0, TH) == "fallback"
