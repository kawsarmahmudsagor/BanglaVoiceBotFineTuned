"""`check-eval-text`: find domain eval sentences that already occur in a training corpus (spec §10.2)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from app.core.text import bn_normalize

from .workdir import StepFile, WorkDir


def read_sentences(path: Path) -> list[tuple[int, str]]:
    """(line number, sentence) for every line that is not blank and not a `#` comment."""
    out = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if line and not line.startswith("#"):
            out.append((n, line))
    return out


def unmatchable(sentences: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Sentences whose normalized form is empty (punctuation only): skipped, they would match every empty clip."""
    return [(n, s) for n, s in sentences if not bn_normalize(s, mode="asr")]


def find_matches(sentences: list[tuple[int, str]], work_dirs: list[Path]) -> list[tuple[int, str, Path, str]]:
    """(line, sentence, work dir, clip id) for each clip whose normalized text_clean equals a sentence's.

    Sentences that normalize to an empty string are skipped (see `unmatchable`)."""
    wanted: dict[str, list[tuple[int, str]]] = {}
    for n, s in sentences:
        key = bn_normalize(s, mode="asr")
        if key:
            wanted.setdefault(key, []).append((n, s))
    matches = []
    for root in work_dirs:
        df: pd.DataFrame = StepFile.read(WorkDir(root).step_path("text"), step="text", version=1)
        for rid, text in zip(df["id"], df["text_clean"].fillna("")):
            for n, s in wanted.get(bn_normalize(text, mode="asr"), []):
                matches.append((n, s, Path(root), rid))
    return sorted(matches, key=lambda m: (m[0], str(m[2]), m[3]))


def _cmd(args: argparse.Namespace) -> int:
    sentences = read_sentences(args.sentences)
    matches = find_matches(sentences, args.against)
    skipped = unmatchable(sentences)
    if skipped:
        print(f"skipped {len(skipped)} line(s) that normalize to nothing: {', '.join(str(n) for n, _ in skipped)}")
    for n, s, root, rid in matches:
        print(f"line {n}: {s}  <- {root} clip {rid}")
    print(f"{len({m[0] for m in matches})} of {len(sentences)} sentences found in the training text")
    return 1 if matches else 0


def add_parser(commands) -> None:
    p = commands.add_parser("check-eval-text", help="report eval sentences that match training transcripts (spec 10.2)")
    p.add_argument("--sentences", type=Path, required=True, help="text file, one sentence per line (# comments and blank lines skipped)")
    p.add_argument("--against", type=Path, action="append", required=True, metavar="WORK_DIR",
                   help="work dir whose text.parquet is checked; repeatable")
    p.set_defaults(func=_cmd)
