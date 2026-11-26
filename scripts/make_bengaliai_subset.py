"""Build a random Bengali.AI subset zip (metadata_train.csv + metadata_valid.csv + mp3s) for dataset upload.

Default ~160k train clips (~200 h at 4.5 s avg) + 3k valid clips from the official valid split.
Usage: python scripts/make_bengaliai_subset.py [--train 160000] [--valid 3000] [--out bengaliai_200h.zip]
"""
import argparse
import csv
import io
import random
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path.home() / ".cache/kagglehub/competitions/bengaliai-speech"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=160_000)
    ap.add_argument("--valid", type=int, default=3_000)
    ap.add_argument("--max-dup", type=int, default=2, help="max clips sharing one sentence")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="bengaliai_200h.zip")
    a = ap.parse_args()

    rows: dict[str, list[tuple[str, str]]] = {"train": [], "valid": []}
    with open(ROOT / "train.csv", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            text = unicodedata.normalize("NFC", r["sentence"]).strip()
            if 3 <= len(text) <= 300 and r["split"] in rows:
                rows[r["split"]].append((r["id"], text))

    rng = random.Random(a.seed)
    picked: dict[str, list[tuple[str, str]]] = {}
    seen_valid = set()
    for split, n in (("valid", a.valid), ("train", a.train)):
        rng.shuffle(rows[split])
        dup, out = Counter(), []
        for i, t in rows[split]:
            # train must not share sentences with valid
            if split == "train" and t in seen_valid:
                continue
            if dup[t] >= (1 if split == "valid" else a.max_dup):
                continue
            dup[t] += 1
            out.append((i, t))
            if len(out) == n:
                break
        picked[split] = out
        if split == "valid":
            seen_valid = {t for _, t in out}

    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_STORED) as zf:
        for split, items in picked.items():
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["file", "text"])
            w.writerows((f"clips/{i}.mp3", t) for i, t in items)
            zf.writestr(f"metadata_{split}.csv", buf.getvalue().encode("utf-8"))
            for i, _ in items:
                zf.write(ROOT / "train_mp3s" / f"{i}.mp3", f"clips/{i}.mp3")
    print({k: len(v) for k, v in picked.items()}, "->", a.out)


if __name__ == "__main__":
    main()
