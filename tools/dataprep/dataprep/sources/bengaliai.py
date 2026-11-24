"""Bengali.AI speech corpus (Kaggle layout): train.csv + train_mp3s/<id>.mp3, read in place."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from app.core.splits import normalize_split

from ..workdir import StepFile, WorkDir

DEFAULT_SRC = "~/.cache/kagglehub/competitions/bengaliai-speech"


def import_bengaliai(src: Path, wd: WorkDir) -> int:
    """Write clips.parquet (id, src_path, text_raw, src_split); return the number of clips imported."""
    src = Path(src)
    csv_path, audio_dir = src / "train.csv", src / "train_mp3s"
    if not csv_path.is_file() or not audio_dir.is_dir():
        raise SystemExit(f"{src} is not a Bengali.AI dir: expected train.csv and train_mp3s/ inside it")
    meta = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    rows, missing = [], 0
    for rid, sentence, split in zip(meta["id"], meta["sentence"], meta["split"]):
        audio = audio_dir / f"{rid}.mp3"
        if not audio.is_file():
            missing += 1
            continue
        rows.append(
            {"id": rid, "src_path": str(audio.resolve()), "text_raw": sentence, "src_split": normalize_split(split)}
        )
    df = pd.DataFrame(rows, columns=["id", "src_path", "text_raw", "src_split"])
    StepFile.write(wd.clips_path, df, step="import", version=1, params={"source": "bengaliai"})
    print(f"imported {len(df)} clips, {missing} missing audio skipped")
    return len(df)


def _cmd(args: argparse.Namespace) -> int:
    src = args.src or Path(os.environ.get("BENGALIAI_SRC") or DEFAULT_SRC).expanduser()
    import_bengaliai(src, WorkDir(args.work))
    return 0


def add_parser(import_commands) -> None:
    p = import_commands.add_parser("bengaliai", help="import the Bengali.AI Kaggle corpus in place")
    p.add_argument("--src", type=Path, help=f"corpus dir (default: $BENGALIAI_SRC, else {DEFAULT_SRC})")
    p.add_argument("--work", type=Path, default=Path("var/dataprep/bengaliai"), help="work dir")
    p.set_defaults(func=_cmd)
