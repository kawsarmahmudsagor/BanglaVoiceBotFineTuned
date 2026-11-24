"""`text` step: clean transcripts (spec 7.1) and compute flags. Stored text is cleaned, never bn_normalized."""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

from app.core.text import _QUOTES, _strip_zero_width, bn_normalize

from ..workdir import WorkDir
from ._runner import run_step

STEP = "text"
VERSION = 2
COLUMNS = ["id", "text_clean", "has_latin", "has_digits", "n_chars", "empty_asr", "error"]

_normalizer = None
_BN_RUN = re.compile("[ঀ-৿‌‍]+")


def _bn_fix(text: str) -> str:
    """bnunicodenormalizer per word, applied to each Bengali run so attached punctuation/Latin survives
    (given a whole word it drops leading quotes); runs it cannot repair stay as they are."""
    global _normalizer
    if _normalizer is None:
        from bnunicodenormalizer import Normalizer

        _normalizer = Normalizer()
    # The library returns None for a run it judges unrepairable (e.g. a lone vowel sign); that run is kept as is.
    def fix(piece: str) -> str:
        return (_normalizer(piece)["normalized"] or piece) if piece else piece

    return _BN_RUN.sub(lambda m: _fix_masked(m[0], fix), text)


# A ZWJ/ZWNJ next to a hasanta, between two Bengali characters: it selects a conjunct form (র‍্য, র্‍য, ক্‌ষ), and
# app.core.text._strip_zero_width keeps it there (a joiner between two Bengali characters). bnunicodenormalizer
# strips such joiners (and the hasanta with them in ক্‌ষ), so these hasanta+joiner pairs are masked out of its
# input and restored. The library also drops any placeholder character, so the mask splits the run instead.
_HASANTA_JOINER = re.compile("(্[‌‍](?=[ঀ-৿])|(?<=[ঀ-৿])[‌‍]্)")


def _fix_masked(run: str, fix) -> str:
    parts = _HASANTA_JOINER.split(run)  # odd indices are the masked hasanta+joiner pairs
    return "".join(p if i % 2 else fix(p) for i, p in enumerate(parts))


def clean(text: str) -> str:
    """Spec 7.1 order: bnunicodenormalizer, NFC, zero-width removal, danda/quote unification, whitespace.

    app.core.text has no danda/quote helper, so `|` -> `।` and the quote table (_QUOTES, imported) are applied here.
    """
    text = _bn_fix(text)
    text = unicodedata.normalize("NFC", text)
    text = _strip_zero_width(text)
    text = text.translate(_QUOTES).replace("|", "।")
    return re.sub(r"\s+", " ", text).strip()


def _row(row: dict) -> dict:
    text = clean(row["text_raw"])
    asr = bn_normalize(text, mode="asr")
    n_chars = len(asr.replace(" ", ""))
    return {
        "id": row["id"],
        "text_clean": text,
        "has_latin": bool(re.search(r"[A-Za-z]", text)),
        "has_digits": bool(re.search(r"[0-9০-৯]", text)),
        "n_chars": n_chars,
        "empty_asr": n_chars == 0,
        "error": None,
    }


def run(wd: WorkDir, *, workers: int = 1, force: bool = False, retry_errors: bool = False) -> int:
    return run_step(
        wd, step=STEP, version=VERSION, columns=COLUMNS, row_fn=_row, input_cols=["id", "text_raw"],
        dtypes={"has_latin": "boolean", "has_digits": "boolean", "empty_asr": "boolean", "n_chars": "Int64"},
        workers=workers, force=force, retry_errors=retry_errors,
    )


def _cmd(args: argparse.Namespace) -> int:
    run(WorkDir(args.work), workers=args.workers, force=args.force, retry_errors=args.retry_errors)
    return 0


def add_parser(commands) -> None:
    p = commands.add_parser(STEP, help="clean transcripts and compute text flags")
    p.add_argument("--work", type=Path, default=Path("var/dataprep/bengaliai"), help="work dir")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--force", action="store_true", help="recompute all rows")
    p.add_argument("--retry-errors", action="store_true", help="recompute rows whose error is set")
    p.set_defaults(func=_cmd)
