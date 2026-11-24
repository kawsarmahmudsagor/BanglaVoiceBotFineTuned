"""`text` step: clean transcripts (spec 7.1) and compute flags. Stored text is cleaned, never bn_normalized."""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd
from app.core.text import _QUOTES, _strip_zero_width, bn_normalize

from ..pool import run_rows
from ..workdir import StepFile, WorkDir

STEP = "text"
VERSION = 1
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
    return _BN_RUN.sub(lambda m: _normalizer(m[0])["normalized"] or m[0], text)


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
    clips = StepFile.read(wd.clips_path, step="import", version=1)
    path = wd.step_path(STEP)
    prev = pd.DataFrame(columns=COLUMNS)
    if path.exists() and not force:
        prev = StepFile.read(path, step=STEP, version=VERSION)
        keep = set(prev["id"]) if retry_errors is False else set(prev.loc[prev["error"].isna(), "id"])
        todo = clips[~clips["id"].isin(keep)]
        prev = prev[prev["id"].isin(keep)]
    else:
        todo = clips
    new = run_rows(_row, todo[["id", "text_raw"]].to_dict("records"), workers=workers, desc="text")
    new_df = pd.DataFrame(new).reindex(columns=COLUMNS)
    out = new_df if prev.empty else pd.concat([prev, new_df], ignore_index=True)
    out = out.astype({"has_latin": "boolean", "has_digits": "boolean", "empty_asr": "boolean", "n_chars": "Int64"})
    StepFile.write(path, out, step=STEP, version=VERSION, params={})
    return len(new_df)


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
