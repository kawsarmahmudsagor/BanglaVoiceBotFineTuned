"""Merge OpenSLR 53 (Large Bengali ASR) shard zips into one ASR dataset zip for upload / local-path import.

Each shard asr_bengali_<h>.zip holds asr_bengali/data/<h?>/<id>.flac plus the full utt_spk_text.tsv
(id, speaker, text; no header). Only clips present in the given shards are kept, so any subset of shards works.

Output zip (stored, FLAC is already compressed):
  metadata.csv   file,text,split,speaker   split = train | valid | test
  data/<xx>/<id>.flac
  LICENSE        (CC BY-SA 4.0, from the corpus)

valid/test: whole speakers picked by sha1 (stable when shards are added later), so no speaker is in two splits;
use `use_dataset_splits: true` in the recipe. Train clips whose sentence also appears in valid/test are dropped
so held-out WER is not inflated by memorised sentences. Text is NFC + whitespace-collapsed only; the Whisper
backend applies bn_normalize and resamples to 16 kHz itself.

Usage: python scripts/make_openslr53_dataset.py SHARD_DIR_OR_ZIPS... [--out openslr53_bn.zip]
       [--valid-frac 0.05] [--test-frac 0.05]
"""
import argparse
import csv
import hashlib
import io
import re
import shutil
import sys
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

_ENTRY_RE = re.compile(r"^asr_bengali/data/([0-9a-f]{2})/([0-9a-f]+)\.flac$")
_WS_RE = re.compile(r"\s+")


def _shard_zips(inputs: list[str]) -> list[Path]:
    zips: list[Path] = []
    for s in inputs:
        p = Path(s)
        zips += sorted(p.glob("asr_bengali_*.zip")) if p.is_dir() else [p]
    if not zips:
        sys.exit("no asr_bengali_*.zip found")
    return zips


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def _read_tsv(raw: bytes) -> dict[str, tuple[str, str]]:
    rows: dict[str, tuple[str, str]] = {}
    for line in raw.decode("utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0].strip():
            rows[parts[0].strip()] = (parts[1].strip(), _clean("\t".join(parts[2:])))
    return rows


def _split_of(speaker: str, valid_frac: float, test_frac: float) -> str:
    u = int(hashlib.sha1(speaker.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "valid" if u < valid_frac else "test" if u < valid_frac + test_frac else "train"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="folder with asr_bengali_*.zip, or the zip files")
    ap.add_argument("--out", default="openslr53_bn.zip")
    ap.add_argument("--valid-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    a = ap.parse_args()

    zips = _shard_zips(a.inputs)
    tsv: dict[str, tuple[str, str]] | None = None
    tsv_hash = None
    license_bytes = None
    clips: dict[str, tuple[Path, str]] = {}  # id -> (shard zip, member name)
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
            raw = zf.read("asr_bengali/utt_spk_text.tsv")
            h = hashlib.sha256(raw).hexdigest()
            if tsv is None:
                tsv, tsv_hash = _read_tsv(raw), h
            elif h != tsv_hash:
                print(f"note: {z.name} has a different utt_spk_text.tsv; merging rows")
                for k, v in _read_tsv(raw).items():
                    if k in tsv and tsv[k] != v:
                        sys.exit(f"conflicting transcript rows for '{k}' across shards")
                    tsv[k] = v
            if license_bytes is None and "asr_bengali/LICENSE" in names:
                license_bytes = zf.read("asr_bengali/LICENSE")
            n = 0
            for name in names:
                m = _ENTRY_RE.match(name)
                if m and m.group(2) not in clips:
                    clips[m.group(2)] = (z, name)
                    n += 1
            print(f"{z.name}: {n} clips")
    assert tsv is not None

    no_row = sorted(i for i in clips if i not in tsv)
    empty = sorted(i for i in clips if i in tsv and not tsv[i][1])
    keep = sorted(i for i in clips if i in tsv and tsv[i][1])
    split = {i: _split_of(tsv[i][0], a.valid_frac, a.test_frac) for i in keep}
    valid = [i for i in keep if split[i] == "valid"]
    test = [i for i in keep if split[i] == "test"]
    held_text = {tsv[i][1] for i in valid + test}
    train_all = [i for i in keep if split[i] == "train"]
    train = [i for i in train_all if tsv[i][1] not in held_text]

    out = Path(a.out)
    tmp = out.with_suffix(out.suffix + ".part")
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["file", "text", "split", "speaker"])
    rel = {i: clips[i][1].removeprefix("asr_bengali/") for i in keep}
    for name, ids in (("train", train), ("valid", valid), ("test", test)):
        for i in ids:
            w.writerow([rel[i], tsv[i][1], name, tsv[i][0]])
    by_zip: dict[Path, list[str]] = {}
    for i in train + valid + test:
        by_zip.setdefault(clips[i][0], []).append(i)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as zo:
        zo.writestr("metadata.csv", buf.getvalue().encode("utf-8"))
        if license_bytes:
            zo.writestr("LICENSE", license_bytes)
        for z, ids in by_zip.items():
            with zipfile.ZipFile(z) as zf:
                for i in ids:
                    with zf.open(clips[i][1]) as src, zo.open(rel[i], "w", force_zip64=True) as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
            print(f"copied {len(ids)} clips from {z.name}")
    tmp.replace(out)

    spk = Counter(tsv[i][0] for i in train + valid + test)
    vspk = {tsv[i][0] for i in valid}
    tspk = {tsv[i][0] for i in test}
    print(f"\nshards: {len(zips)}  clips in zips: {len(clips)}  tsv rows: {len(tsv)}")
    print(f"dropped: {len(no_row)} without a transcript row, {len(empty)} with empty text, "
          f"{len(train_all) - len(train)} train clips sharing a sentence with valid/test")
    print(f"train: {len(train)}  valid: {len(valid)}  test: {len(test)}  "
          f"speakers: {len(spk)} ({len(vspk)} valid, {len(tspk)} test)")
    print(f"-> {out}  ({out.stat().st_size / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
