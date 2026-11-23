"""Rule-based metrics for eval loops and LLM reply checks (spec section 8).

Pure Python, no third-party dependencies. Callers normalize text before scoring.
"""

import re
import unicodedata
from collections.abc import Sequence

_BANGLA_START = 0x0980
_BANGLA_END = 0x09FF

_EMOJI_RANGES = (
    (0x1F300, 0x1FAFF),
    (0x2600, 0x27BF),
    (0x1F600, 0x1F64F),
    (0x1F680, 0x1F6FF),
)

_MARKDOWN_PATTERNS = (
    re.compile(r"^\s{0,3}#{1,6}\s+\S", re.MULTILINE),  # heading
    re.compile(r"^\s*[-*+]\s+\S", re.MULTILINE),  # bullet list item
    re.compile(r"\*\*[^*\n]+\*\*"),  # bold
    re.compile(r"`[^`\n]+`"),  # inline code
    re.compile(r"\[[^\]\n]+\]\([^)\n]+\)"),  # link
)


def _edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Levenshtein distance over token or character sequences, O(len(a) * len(b))."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, start=1):
            cost = 0 if x == y else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def _corpus_rate(refs: list[str], hyps: list[str], units) -> float:
    if len(refs) != len(hyps):
        raise ValueError(f"refs and hyps length differ: {len(refs)} != {len(hyps)}")
    edits = 0
    total_ref = 0
    for ref, hyp in zip(refs, hyps):
        r = units(ref)
        edits += _edit_distance(r, units(hyp))
        total_ref += len(r)
    return edits / max(1, total_ref)


def _words(text: str) -> list[str]:
    return text.split()


def _chars(text: str) -> list[str]:
    return list("".join(text.split()))


def wer(refs: list[str], hyps: list[str]) -> float:
    """Corpus word error rate: total word edits divided by total reference words."""
    return _corpus_rate(refs, hyps, _words)


def cer(refs: list[str], hyps: list[str]) -> float:
    """Corpus character error rate over non-whitespace characters."""
    return _corpus_rate(refs, hyps, _chars)


def sample_wer(ref: str, hyp: str) -> float:
    """WER for one reference/hypothesis pair."""
    return wer([ref], [hyp])


def _is_letter(c: str) -> bool:
    if c.isalpha():
        return True
    return _BANGLA_START <= ord(c) <= _BANGLA_END and unicodedata.category(c).startswith("M")


def bangla_script_share(text: str) -> float:
    """Bangla letters divided by all letters; 0.0 when the text has no letters.

    A letter is ``str.isalpha()`` or a Bangla combining mark (vowel signs count as Bangla text).
    """
    letters = [c for c in text if _is_letter(c)]
    if not letters:
        return 0.0
    bangla = sum(1 for c in letters if _BANGLA_START <= ord(c) <= _BANGLA_END)
    return bangla / len(letters)


def has_markdown(text: str) -> bool:
    """True if the text contains common Markdown syntax (headings, lists, bold, code, links)."""
    return any(p.search(text) for p in _MARKDOWN_PATTERNS)


def has_emoji(text: str) -> bool:
    """True if any code point falls in the common emoji blocks."""
    return any(lo <= ord(c) <= hi for c in text for lo, hi in _EMOJI_RANGES)
