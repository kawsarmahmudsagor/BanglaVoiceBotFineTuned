"""Bangla text normalization shared by ASR/TTS data converters (spec section 5).

``mode="tts"`` expands numbers, dates, times, currency and percentages into words.
``mode="asr"`` produces scoring text: lowercase Latin, Bengali digits, no punctuation.
"""

import re
import unicodedata
from typing import Literal

BN_NORMALIZE_VERSION: int = 1

_WORDS_0_99 = {
    0: "শূন্য", 1: "এক", 2: "দুই", 3: "তিন", 4: "চার", 5: "পাঁচ", 6: "ছয়", 7: "সাত", 8: "আট", 9: "নয়",
    10: "দশ", 11: "এগারো", 12: "বারো", 13: "তেরো", 14: "চৌদ্দ", 15: "পনেরো", 16: "ষোলো", 17: "সতেরো",
    18: "আঠারো", 19: "উনিশ",
    20: "বিশ", 21: "একুশ", 22: "বাইশ", 23: "তেইশ", 24: "চব্বিশ", 25: "পঁচিশ", 26: "ছাব্বিশ", 27: "সাতাশ",
    28: "আঠাশ", 29: "ঊনত্রিশ",
    30: "ত্রিশ", 31: "একত্রিশ", 32: "বত্রিশ", 33: "তেত্রিশ", 34: "চৌত্রিশ", 35: "পঁয়ত্রিশ", 36: "ছত্রিশ",
    37: "সাঁইত্রিশ", 38: "আটত্রিশ", 39: "ঊনচল্লিশ",
    40: "চল্লিশ", 41: "একচল্লিশ", 42: "বিয়াল্লিশ", 43: "তেতাল্লিশ", 44: "চুয়াল্লিশ", 45: "পঁয়তাল্লিশ",
    46: "ছেচল্লিশ", 47: "সাতচল্লিশ", 48: "আটচল্লিশ", 49: "ঊনপঞ্চাশ",
    50: "পঞ্চাশ", 51: "একান্ন", 52: "বায়ান্ন", 53: "তিপ্পান্ন", 54: "চুয়ান্ন", 55: "পঞ্চান্ন", 56: "ছাপ্পান্ন",
    57: "সাতান্ন", 58: "আটান্ন", 59: "ঊনষাট",
    60: "ষাট", 61: "একষট্টি", 62: "বাষট্টি", 63: "তেষট্টি", 64: "চৌষট্টি", 65: "পঁয়ষট্টি", 66: "ছেষট্টি",
    67: "সাতষট্টি", 68: "আটষট্টি", 69: "ঊনসত্তর",
    70: "সত্তর", 71: "একাত্তর", 72: "বাহাত্তর", 73: "তিয়াত্তর", 74: "চুয়াত্তর", 75: "পঁচাত্তর", 76: "ছিয়াত্তর",
    77: "সাতাত্তর", 78: "আটাত্তর", 79: "ঊনআশি",
    80: "আশি", 81: "একাশি", 82: "বিরাশি", 83: "তিরাশি", 84: "চুরাশি", 85: "পঁচাশি", 86: "ছিয়াশি",
    87: "সাতাশি", 88: "আটাশি", 89: "ঊননব্বই",
    90: "নব্বই", 91: "একানব্বই", 92: "বিরানব্বই", 93: "তিরানব্বই", 94: "চুরানব্বই", 95: "পঁচানব্বই",
    96: "ছিয়ানব্বই", 97: "সাতানব্বই", 98: "আটানব্বই", 99: "নিরানব্বই",
}  # fmt: skip
_MONTHS = [
    "জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন",
    "জুলাই", "আগস্ট", "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর",
]  # fmt: skip

_BN_DIGITS = "০১২৩৪৫৬৭৮৯"
_LATIN_TO_BN = str.maketrans("0123456789", _BN_DIGITS)
_BN_TO_LATIN = str.maketrans(_BN_DIGITS, "0123456789")
_D = "[০-৯]"
_ZW = {"​", "‌", "‍", "⁠", "﻿"}
_QUOTES = str.maketrans({"“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"', "‘": "'", "’": "'", "‚": "'"})


def number_to_bangla_words(n: int) -> str:
    """Spell a non-negative integer in Bangla (lakh/crore system)."""
    if n < 0:
        raise ValueError("negative numbers are not supported")
    if n < 100:
        return _WORDS_0_99[n]
    for unit, name in ((10_000_000, "কোটি"), (100_000, "লাখ"), (1000, "হাজার")):
        if n >= unit:
            head, rest = divmod(n, unit)
            out = f"{number_to_bangla_words(head)} {name}"
            return f"{out} {number_to_bangla_words(rest)}" if rest else out
    head, rest = divmod(n, 100)
    out = f"{_WORDS_0_99[head]}শো"
    return f"{out} {_WORDS_0_99[rest]}" if rest else out


def _int(s: str) -> int:
    return int(s.translate(_BN_TO_LATIN))


def _decimal_words(s: str) -> str:
    whole, _, frac = s.partition(".")
    out = number_to_bangla_words(_int(whole))
    if frac:
        out += " দশমিক " + " ".join(_WORDS_0_99[int(c)] for c in frac.translate(_BN_TO_LATIN))
    return out


def _date(m: re.Match) -> str:
    day, month, year = _int(m[1]), _int(m[2]), _int(m[3])
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return m[0]
    return f"{number_to_bangla_words(day)} {_MONTHS[month - 1]} {number_to_bangla_words(year)}"


def _time(m: re.Match) -> str:
    hour, minute = _int(m[1]), _int(m[2])
    if hour > 23 or minute > 59:
        return m[0]
    out = f"{number_to_bangla_words(hour)}টা"
    return f"{out} {number_to_bangla_words(minute)} মিনিট" if minute else out


def _strip_zero_width(text: str) -> str:
    out = []
    for i, ch in enumerate(text):
        if ch in _ZW:
            joiner = ch in ("‌", "‍")
            prev_bn = i > 0 and "ঀ" <= text[i - 1] <= "৿"
            next_bn = i + 1 < len(text) and "ঀ" <= text[i + 1] <= "৿"
            if joiner and prev_bn and next_bn:
                out.append(ch)
            continue
        out.append(ch)
    return "".join(out)


def _expand_tts(text: str) -> str:
    text = re.sub(rf"(?<={_D}),(?={_D}{{3}})", "", text)
    text = re.sub(rf"(?<!{_D})({_D}{{1,2}})/({_D}{{1,2}})/({_D}{{4}})(?!{_D})", _date, text)
    text = re.sub(rf"(?<!{_D})({_D}{{1,2}}):({_D}{{2}})(?!{_D})", _time, text)
    text = re.sub(rf"৳\s*({_D}+(?:\.{_D}+)?)", lambda m: _decimal_words(m[1]) + " টাকা", text)
    text = re.sub(rf"({_D}+(?:\.{_D}+)?)\s*%", lambda m: _decimal_words(m[1]) + " শতাংশ", text)
    return re.sub(rf"{_D}+(?:\.{_D}+)?", lambda m: _decimal_words(m[0]), text)


def _strip_punct(text: str) -> str:
    return "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)


def bn_normalize(text: str, *, mode: Literal["asr", "tts"]) -> str:
    if mode not in ("asr", "tts"):
        raise ValueError(f"unknown mode: {mode!r}")
    text = unicodedata.normalize("NFC", text)
    text = _strip_zero_width(text)
    text = text.translate(_QUOTES).replace("|", "।")
    text = text.translate(_LATIN_TO_BN)
    if mode == "asr":
        text = _strip_punct(text.lower())
    else:
        text = _expand_tts(text)
    return re.sub(r"\s+", " ", text).strip()
