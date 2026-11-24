"""Stage-B voice set: metadata.csv (file,text[,category,session,...]) + wavs/, read in place (spec §9.1).

clips.parquet gets `id, src_path, text_raw, src_split` (src_split empty: the split is assigned at export,
spec §9.4) plus the optional `category` and `session` columns when metadata.csv has them. The import step
params record `{"source": "voice"}`; export reads it to decide the default provenance.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..workdir import StepFile, WorkDir

OPTIONAL_COLUMNS = ("category", "session")


def _resolve(src: Path, name: str) -> Path | None:
    for cand in (src / name, src / "wavs" / name, src / "wavs" / f"{name}.wav"):
        if cand.is_file():
            return cand
    return None


def import_voice(src: Path, wd: WorkDir) -> int:
    src = Path(src)
    csv_path = src / "metadata.csv"
    if not csv_path.is_file() or not (src / "wavs").is_dir():
        raise SystemExit(f"{src} is not a voice dir: expected metadata.csv and wavs/ inside it")
    meta = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    for col in ("file", "text"):
        if col not in meta.columns:
            raise SystemExit(f"{csv_path} must have columns file,text (found {list(meta.columns)})")
    extras = [c for c in OPTIONAL_COLUMNS if c in meta.columns]
    rows, missing, seen = [], 0, set()
    for rec in meta.to_dict("records"):
        rid = Path(rec["file"]).stem
        if rid in seen:
            raise SystemExit(f"{csv_path}: duplicate take id '{rid}'")
        seen.add(rid)
        audio = _resolve(src, rec["file"])
        if audio is None:
            missing += 1
            continue
        rows.append({"id": rid, "src_path": str(audio.resolve()), "text_raw": rec["text"], "src_split": "",
                     **{c: rec[c] for c in extras}})
    df = pd.DataFrame(rows, columns=["id", "src_path", "text_raw", "src_split", *extras])
    StepFile.write(wd.clips_path, df, step="import", version=1, params={"source": "voice"})
    print(f"imported {len(df)} clips, {missing} missing audio skipped")
    return len(df)


def _cmd(args: argparse.Namespace) -> int:
    import_voice(args.src, WorkDir(args.work))
    return 0


def add_parser(import_commands) -> None:
    p = import_commands.add_parser("voice", help="import the recorded stage-B voice set (metadata.csv + wavs/) in place")
    p.add_argument("--src", type=Path, required=True, help="dir with metadata.csv and wavs/")
    p.add_argument("--work", type=Path, default=Path("var/dataprep/voice"), help="work dir")
    p.set_defaults(func=_cmd)
