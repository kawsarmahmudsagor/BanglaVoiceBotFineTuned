"""Tiny on-disk fixtures shared by the dataprep tests."""
from __future__ import annotations

import csv
from pathlib import Path


def make_voice(src: Path, rows: list[dict], *, sr: int = 48000, subtype: str = "PCM_24", seconds: float = 1.0) -> Path:
    """Stage-B voice layout: metadata.csv (file,text[,category,session,...]) and wavs/<name>.wav.

    rows are dicts with `name`, `text` and optional extra columns; row["sr"] / row["amp"] override the defaults.
    """
    import numpy as np
    import soundfile as sf

    (src / "wavs").mkdir(parents=True, exist_ok=True)
    extra = [k for k in rows[0] if k not in ("name", "text", "sr", "amp")] if rows else []
    with open(src / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "text", *extra])
        for r in rows:
            w.writerow([f"wavs/{r['name']}.wav", r["text"], *(r[k] for k in extra)])
            rate = r.get("sr", sr)
            t = np.arange(int(rate * seconds)) / rate
            y = np.clip(r.get("amp", 0.3) * np.sin(2 * np.pi * 220 * t), -1, 1)
            sf.write(src / "wavs" / f"{r['name']}.wav", y, rate, subtype=subtype)
    return src


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
