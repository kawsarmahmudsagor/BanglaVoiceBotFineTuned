"""Tiny on-disk fixtures shared by the dataprep tests."""
from __future__ import annotations

import csv
from pathlib import Path


def make_bengaliai(src: Path, rows: list[tuple[str, str, str]], *, with_audio: set[str] | None = None) -> Path:
    """Kaggle Bengali.AI layout: train.csv (id,sentence,split) and train_mp3s/<id>.mp3.

    rows are (id, sentence, split). Audio exists for every id unless with_audio narrows it.
    """
    (src / "train_mp3s").mkdir(parents=True, exist_ok=True)
    with open(src / "train.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "sentence", "split"])
        w.writerows(rows)
    for rid, _, _ in rows:
        if with_audio is None or rid in with_audio:
            (src / "train_mp3s" / f"{rid}.mp3").write_bytes(b"\x00")
    return src
