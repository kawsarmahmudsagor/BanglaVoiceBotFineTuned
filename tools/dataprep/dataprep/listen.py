"""`listen`: a static HTML listening sheet with a few clips per score band (spec 5.2 step 4).

Clips are those that pass the profile's filters, ignoring the filter on the listened column, so the
bands show what a threshold on that column would keep or drop. Within a band, clips are taken in a
deterministic hash order, so reruns give the same sheet.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import os
from pathlib import Path

import pandas as pd

from .profiles import Profile, load_profile
from .roundtrip_cosyvoice import _REPORT_CSS
from .select import COLUMN_STEPS, SelectError, eligible, resolve_skips
from .workdir import StepFile, WorkDir

DEFAULT_BANDS = [0, 2.5, 2.8, 3.0, 3.2, 3.5, 5]
# Scores shown next to each clip when their step file exists.
SHOWN = ["dnsmos_p808", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "snr_db", "c50_db", "max_pause_s"]

_EXTRA_CSS = "td.score { font-variant-numeric: tabular-nums; white-space: nowrap; } h2 { font-size:1.1rem; margin-top:28px; }"


def _hash_key(clip_id: str) -> str:
    return hashlib.sha1(clip_id.encode("utf-8")).hexdigest()


def _src(html_dir: Path, audio: Path) -> str:
    try:
        return Path(os.path.relpath(audio, html_dir)).as_posix()
    except ValueError:  # different drive on Windows
        return audio.resolve().as_uri()


def _read_scores(wd: WorkDir, column: str) -> pd.DataFrame:
    steps: dict[str, int] = {}
    for col in [column, *SHOWN]:
        if col in COLUMN_STEPS:
            steps[COLUMN_STEPS[col][0]] = COLUMN_STEPS[col][1]
    frames = []
    for step, version in steps.items():
        path = wd.step_path(step)
        if not path.exists():
            if step == COLUMN_STEPS[column][0]:
                raise SelectError(f"listen needs step '{step}' for {column} but {path} does not exist; run that step first")
            continue
        df = StepFile.read(path, step=step, version=version, params=None)
        frames.append(df[["id", *[c for c in SHOWN if c in df.columns]]])
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="id", how="outer")
    return out


def _band_label(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g}"


def listen(
    wd: WorkDir,
    profile: Profile,
    *,
    column: str = "dnsmos_ovrl",
    per_band: int = 10,
    bands: list[float] = DEFAULT_BANDS,
    labeller: str | None = None,
) -> Path:
    if column not in SHOWN:
        raise SelectError(f"cannot listen by '{column}'; choose one of: {', '.join(SHOWN)}")
    skip, notes = resolve_skips(wd, profile, labeller, frozenset({column}), speaker_if_missing=True)
    for note in notes:
        print(f"listen: {note}")
    ok = eligible(wd, profile, labeller, skip_columns=skip)[["id", "src_split"]]
    clips = StepFile.read(wd.clips_path, step="import", version=1)[["id", "src_path", "text_raw"]]
    df = ok.merge(clips, on="id").merge(_read_scores(wd, column), on="id", how="left")
    df = df[df[column].notna()].copy()
    df["_h"] = df["id"].map(_hash_key)

    out_path = wd.root / "listen" / f"{profile.name}.html"
    html_dir = out_path.parent
    html_dir.mkdir(parents=True, exist_ok=True)
    shown = [c for c in SHOWN if c in df.columns]

    sections = []
    for i, (lo, hi) in enumerate(zip(bands, bands[1:])):
        last = i == len(bands) - 2
        in_band = df[(df[column] >= lo) & ((df[column] <= hi) if last else (df[column] < hi))]
        picked = in_band.sort_values("_h").head(per_band)
        label = _band_label(lo, hi)
        rows = []
        for r in picked.itertuples(index=False):
            row = r._asdict()
            audio = Path(row["src_path"])
            scores = "".join(
                f"<td class=\"score\">{html.escape(c)} {row[c]:.2f}</td>" if pd.notna(row[c]) else "<td class=\"score\"></td>"
                for c in shown
            )
            rows.append(
                f"<tr data-band=\"{html.escape(label)}\">"
                f"<td>{html.escape(str(row['id']))}</td>"
                f"<td class=\"text\">{html.escape(str(row['text_raw']))}</td>"
                f"<td><audio controls preload=\"none\" src=\"{html.escape(_src(html_dir, audio), quote=True)}\"></audio></td>"
                f"{scores}</tr>"
            )
        sections.append(
            f"<h2>{html.escape(column)} in [{lo:g}, {hi:g}{']' if last else ')'}"
            f" <span class=\"muted\">· {len(in_band)} clips, showing {len(picked)}</span></h2>"
            f"<table><tbody>{''.join(rows)}</tbody></table>"
        )

    name = html.escape(profile.name)
    out_path.write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Listening sheet: {name}</title><style>{_REPORT_CSS}{_EXTRA_CSS}</style></head>
<body>
<h1>Listening sheet: {name}</h1>
<p class="muted">{per_band} clips per {html.escape(column)} band, picked in a fixed hash order from the clips that pass the profile's other filters. Listen, then set the threshold in the profile.</p>
{''.join(sections)}
</body></html>
""",
        encoding="utf-8",
    )
    return out_path


def _cmd(args: argparse.Namespace) -> int:
    try:
        out = listen(WorkDir(args.work), load_profile(args.profile), column=args.column, per_band=args.per_band, labeller=args.labeller)
    except SelectError as e:
        print(f"listen: {e}")
        return 1
    print(out)
    return 0


def add_parser(commands) -> None:
    p = commands.add_parser("listen", help="write a static HTML listening sheet with clips per score band")
    p.add_argument("--work", type=Path, default=Path("var/dataprep/bengaliai"), help="work dir")
    p.add_argument("--profile", required=True, help="profile name or YAML path")
    p.add_argument("--column", default="dnsmos_ovrl", help="score column to band by")
    p.add_argument("--labeller", default=None, help="CER labeller spec; without it the cer filter is skipped")
    p.add_argument("--per-band", type=int, default=10)
    p.set_defaults(func=_cmd)
