import unicodedata

import pytest

from app.core.text import BN_NORMALIZE_VERSION, bn_normalize, number_to_bangla_words

CASES_BOTH = [
    ("কি​ছু", "কিছু"),  # zero-width space removed
    ("র‍যাব", "র‍যাব"),  # ZWJ kept between Bangla letters
    ("a‍b", "ab"),  # ZWJ dropped outside Bangla
    ("আমি   যাব।", "আমি যাব।"),  # whitespace collapsed
    ("“হ্যাঁ”", '"হ্যাঁ"'),  # quotes unified
    ("যাব|", "যাব।"),  # pipe-as-danda unified
]
CASES_ASR = [
    ("Hello, আমি যাব।", "hello আমি যাব"),
    ("১২৩ টাকা", "১২৩ টাকা"),
    ("123", "১২৩"),
]
CASES_TTS = [
    ("১২৩", "একশো তেইশ"),
    ("123", "একশো তেইশ"),
    ("৳৫০০", "পাঁচশো টাকা"),
    ("২৫%", "পঁচিশ শতাংশ"),
    ("১০:৩০", "দশটা ত্রিশ মিনিট"),
    ("০৫/০১/২০২৬", "পাঁচ জানুয়ারি দুই হাজার ছাব্বিশ"),
    ("৩.৫", "তিন দশমিক পাঁচ"),
    ("আমি যাব।", "আমি যাব।"),
]


@pytest.mark.parametrize("src,want", CASES_BOTH)
@pytest.mark.parametrize("mode", ["asr", "tts"])
def test_both(src, want, mode):
    out = bn_normalize(src, mode=mode)
    if mode == "asr":
        want = want.replace("।", "").replace('"', "")
    assert out == want


@pytest.mark.parametrize("src,want", CASES_ASR)
def test_asr(src, want):
    assert bn_normalize(src, mode="asr") == want


@pytest.mark.parametrize("src,want", CASES_TTS)
def test_tts(src, want):
    assert bn_normalize(src, mode="tts") == want


def test_nfc():
    assert bn_normalize("ক়", mode="asr") == unicodedata.normalize("NFC", "ক়")


def test_version():
    assert BN_NORMALIZE_VERSION == 1


def test_number_words_range():
    assert number_to_bangla_words(0) == "শূন্য"
    assert number_to_bangla_words(100000) == "এক লাখ"
    assert number_to_bangla_words(10000000) == "এক কোটি"


@pytest.mark.parametrize(
    "n,want",
    [
        (23, "তেইশ"), (25, "পঁচিশ"), (26, "ছাব্বিশ"), (30, "ত্রিশ"), (11, "এগারো"),
        (29, "ঊনত্রিশ"), (37, "সাঁইত্রিশ"), (52, "বায়ান্ন"), (75, "পঁচাত্তর"),
        (99, "নিরানব্বই"), (100, "একশো"), (200, "দুইশো"), (1000, "এক হাজার"),
        (2026, "দুই হাজার ছাব্বিশ"),
        (123456, "এক লাখ তেইশ হাজার চারশো ছাপ্পান্ন"),
        (25000000, "দুই কোটি পঞ্চাশ লাখ"),
        (250000000, "পঁচিশ কোটি"),
        (10000000000, "এক হাজার কোটি"),
    ],
)
def test_number_words(n, want):
    assert number_to_bangla_words(n) == want
