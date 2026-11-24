"""`check-voice`: QA report for the recorded stage-B voice set (spec §9.3).

Runs `text`, `audio-stats`, `dnsmos`, `brouhaha` and `transcribe` over the imported takes, then writes
`check_voice.json` and `check_voice.html` into the work dir. Per take it flags values outside the `voice`
profile (its `flags` plus its `clip_frac` filter, so the thresholds live only in profiles/voice.yaml) and any
take whose format is not 48 kHz / 24-bit. Per set it reports hours, format consistency, speech rate per
session and phonetic coverage against `data/phonetic_coverage.txt`. A person reviews every flag.
"""
from __future__ import annotations

import argparse
import html
import math
import os
import unicodedata
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from .profiles import Filter, Percentile, Profile, Range, load_profile
from .roundtrip_cosyvoice import _REPORT_CSS
from .select import COLUMN_STEPS, _apply, cer_step_name
from .workdir import StaleStepError, StepFile, WorkDir

DEFAULT_COVERAGE = Path(__file__).resolve().parent.parent / "data" / "phonetic_coverage.txt"
MIN_COVERAGE = 20
EXPECTED_SR = 48000
EXPECTED_BITS = 24
_BITS = {"PCM_U8": 8, "PCM_S8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "FLOAT": 32, "DOUBLE": 64}

# step -> a column it provides; name, version and params come from select.COLUMN_STEPS (one source of truth)
_STEP_COLUMN = {"text": "n_chars", "audio_stats": "dur", "dnsmos": "dnsmos_ovrl", "brouhaha": "snr_db"}


class CheckVoiceError(RuntimeError):
    pass


def load_targets(path: Path | None) -> list[str]:
    """One target per line (NFC); blank lines and `#` comments are skipped; duplicates dropped."""
    if path is None:
        return []
    out: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        t = unicodedata.normalize("NFC", line.strip())
        if t and not t.startswith("#") and t not in out:
            out.append(t)
    return out


def count_target(text: str, target: str) -> int:
    return unicodedata.normalize("NFC", text).count(unicodedata.normalize("NFC", target))


def _flag_filters(profile: Profile) -> list[Filter]:
    return [*profile.flags, *(f for f in profile.filters if f.column == "clip_frac")]


def _limit(f: Filter) -> str:
    if isinstance(f, Range):
        parts = []
        if f.min is not None:
            parts.append(f">= {f.min:g}")
        if f.max is not None:
            parts.append(f"{'<' if f.max_exclusive else '<='} {f.max:g}")
        return " and ".join(parts)
    if isinstance(f, Percentile):
        return f"within p{f.lo:g}-p{f.hi:g}"
    return f"== {f.keep}"


def _num(v) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v) else float(v)


def _read_step(wd: WorkDir, name: str, labeller: str) -> pd.DataFrame:
    if name == "cer":
        step, version, params = cer_step_name(labeller), 1, None
    else:
        step, version, params = COLUMN_STEPS[_STEP_COLUMN[name]]
    path = wd.step_path(step)
    if not path.exists():
        raise CheckVoiceError(f"{path} is missing; run `check-voice` (or the `{step}` step) first")
    df = StepFile.read(path, step=step, version=version, params=params)
    return df.rename(columns={"error": f"error_{name}"})


def _format_of(path: str) -> dict:
    import soundfile as sf

    try:
        info = sf.info(path)
    except Exception as e:  # noqa: BLE001
        return {"sample_rate": None, "bit_depth": None, "label": f"unreadable ({type(e).__name__})"}
    bits = _BITS.get(info.subtype)
    depth = f"{bits}-bit" if bits else info.subtype
    return {"sample_rate": int(info.samplerate), "bit_depth": bits, "label": f"{info.samplerate} Hz / {depth}"}


def _session(clips: pd.DataFrame) -> pd.Series:
    if "session" in clips.columns:
        return clips["session"].map(lambda s: s if s else "unknown")
    return clips["src_path"].map(lambda p: datetime.fromtimestamp(os.path.getmtime(p)).date().isoformat())


def _href(path: str, root: Path) -> str:
    try:
        rel = Path(os.path.relpath(path, root)).as_posix()
    except ValueError:  # another drive: no relative path exists
        rel = Path(path).as_uri()
    return quote(rel, safe="/:._-~")


def build(wd: WorkDir, labeller: str, *, coverage_path: Path | None = DEFAULT_COVERAGE, profile: Profile | None = None) -> dict:
    profile = profile or load_profile("voice")
    clips = StepFile.read(wd.clips_path, step="import", version=1)
    df = clips.copy()
    df["session"] = _session(clips)
    for name in ("text", "audio_stats", "dnsmos", "brouhaha", "cer"):
        df = df.merge(_read_step(wd, name, labeller), on="id", how="left")
    df["cps"] = df["n_chars"].astype("float64") / df["dur"].astype("float64")

    fails: dict[str, list[dict]] = {i: [] for i in df["id"]}
    for f in _flag_filters(profile):
        if f.column not in df.columns:
            raise CheckVoiceError(f"voice profile flags on unknown column '{f.column}'")
        passed = set(_apply(df, f)["id"])
        for rid, value in zip(df["id"], df[f.column]):
            if rid not in passed:
                fails[rid].append({"column": f.column, "value": _num(value), "limit": _limit(f)})

    formats = {rid: _format_of(p) for rid, p in zip(df["id"], df["src_path"])}
    counts: dict[str, int] = {}
    inconsistent = []
    for rid, fmt in formats.items():
        counts[fmt["label"]] = counts.get(fmt["label"], 0) + 1
        if (fmt["sample_rate"], fmt["bit_depth"]) != (EXPECTED_SR, EXPECTED_BITS):
            inconsistent.append({"id": rid, **fmt})
            fails[rid].append({"column": "format", "value": fmt["label"], "limit": f"{EXPECTED_SR} Hz / {EXPECTED_BITS}-bit"})
    err_cols = [c for c in df.columns if c.startswith("error_")]
    for rid, errs in zip(df["id"], df[err_cols].to_dict("records") if err_cols else [{}] * len(df)):
        for col, msg in errs.items():
            if isinstance(msg, str) and msg:
                fails[rid].append({"column": "error", "value": f"{col[6:]}: {msg}", "limit": "no error"})

    by_id = df.set_index("id")
    flagged = [
        {"id": rid, "src_path": by_id.at[rid, "src_path"], "href": _href(by_id.at[rid, "src_path"], wd.root),
         "text": by_id.at[rid, "text_clean"] if isinstance(by_id.at[rid, "text_clean"], str) else by_id.at[rid, "text_raw"],
         "flags": fl}
        for rid, fl in fails.items() if fl
    ]

    sessions = {}
    for name, g in df.groupby("session"):
        sessions[str(name)] = {"n": int(len(g)), "mean_cps": _num(g["cps"].mean())}
    means = [s["mean_cps"] for s in sessions.values() if s["mean_cps"] is not None]
    overall = _num(df["cps"].mean())
    drift = (max(means) - min(means)) / overall if means and overall else 0.0

    targets = load_targets(coverage_path)
    texts = df["text_clean"].fillna("").tolist()
    cov = {t: sum(count_target(x, t) for x in texts) for t in targets}
    return {
        "labeller": labeller,
        "n_takes": int(len(df)),
        "hours": float(df["dur"].sum(skipna=True)) / 3600.0,
        "format": {
            "expected": {"sample_rate": EXPECTED_SR, "bit_depth": EXPECTED_BITS},
            "counts": counts, "consistent": not inconsistent, "inconsistent": inconsistent,
        },
        "speech_rate": {"unit": "n_chars / s", "overall_cps": overall, "sessions": sessions, "drift": drift},
        "coverage": {"min_count": MIN_COVERAGE, "counts": cov, "below": [t for t, n in cov.items() if n < MIN_COVERAGE]},
        "flagged": flagged,
    }


# ---------------------------------------------------------------- html

_EXTRA_CSS = "td.flags { font-size:.9rem; } .chip { display:inline-block; border:1px solid var(--line); border-radius:4px; padding:0 6px; margin:2px 4px 2px 0; }"


def _fmt_value(v) -> str:
    return "missing" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))


def render_flagged_rows(flagged: list[dict]) -> str:
    rows = []
    for t in flagged:
        chips = "".join(
            f'<span class="chip bad">{html.escape(f["column"])} {html.escape(_fmt_value(f["value"]))}'
            f' <span class="muted">(need {html.escape(f["limit"])})</span></span>'
            for f in t["flags"]
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(t['id'])}</td>"
            f"<td><audio controls preload=\"none\" src=\"{html.escape(t['href'], quote=True)}\"></audio></td>"
            f"<td class=\"text\">{html.escape(t['text'])}</td>"
            f"<td class=\"flags\">{chips}</td>"
            "</tr>"
        )
    return "".join(rows)


def render_html(report: dict) -> str:
    fmt, rate, cov = report["format"], report["speech_rate"], report["coverage"]
    fmt_state = '<span class="ok">consistent</span>' if fmt["consistent"] else f'<span class="bad">{len(fmt["inconsistent"])} take(s) differ</span>'
    fmt_counts = ", ".join(f"{html.escape(k)}: {v}" for k, v in fmt["counts"].items())
    sessions = "".join(
        f"<tr><td>{html.escape(s)}</td><td>{v['n']}</td><td>{_fmt_value(v['mean_cps'])}</td></tr>" for s, v in rate["sessions"].items()
    )
    below = "".join(f"<li>{html.escape(t)} ({cov['counts'][t]})</li>" for t in cov["below"])
    below_html = f"<ul>{below}</ul>" if below else '<p class="ok">all targets covered</p>'
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voice set check</title><style>{_REPORT_CSS}{_EXTRA_CSS}</style></head>
<body>
<h1>Voice set check</h1>
<p class="muted">{report['n_takes']} takes · {report['hours']:.2f} h · labeller {html.escape(report['labeller'])}</p>
<h2>Format</h2>
<p>Expected {fmt['expected']['sample_rate']} Hz / {fmt['expected']['bit_depth']}-bit · {fmt_state}</p>
<p class="muted">{fmt_counts}</p>
<h2>Speech rate per session</h2>
<p class="muted">characters per second; drift (max-min over overall mean): {rate['drift']:.1%}</p>
<table><thead><tr><th>Session</th><th>Takes</th><th>Mean chars/s</th></tr></thead><tbody>{sessions}</tbody></table>
<h2>Phonetic coverage</h2>
<p>Targets below {cov['min_count']} occurrences: {len(cov['below'])} of {len(cov['counts'])}</p>
{below_html}
<h2>Flagged takes ({len(report['flagged'])})</h2>
<p class="muted">A person reviews every flag. Re-record or drop flagged takes before export.</p>
<table><thead><tr><th>Take</th><th>Audio</th><th>Text</th><th>Flags</th></tr></thead>
<tbody>{render_flagged_rows(report['flagged'])}</tbody></table>
</body></html>
"""


def write_outputs(wd: WorkDir, report: dict) -> None:
    import json

    wd.root.mkdir(parents=True, exist_ok=True)
    (wd.root / "check_voice.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (wd.root / "check_voice.html").write_text(render_html(report), encoding="utf-8")


# ---------------------------------------------------------------- cli

def run(wd: WorkDir, labeller: str, *, workers: int = 1, coverage_path: Path | None = DEFAULT_COVERAGE) -> dict:
    from .steps import audio_stats, brouhaha, dnsmos, text, transcribe

    text.run(wd, workers=workers)
    audio_stats.run(wd, workers=workers)
    dnsmos.run(wd)
    brouhaha.run(wd)
    transcribe.run(wd, labeller)
    report = build(wd, labeller, coverage_path=coverage_path)
    write_outputs(wd, report)
    return report


def _cmd(args: argparse.Namespace) -> int:
    try:
        report = run(WorkDir(args.work), args.labeller, workers=args.workers, coverage_path=args.coverage)
    except (CheckVoiceError, StaleStepError, RuntimeError, ImportError, ValueError, OSError) as e:
        print(f"check-voice: {e}")
        return 1
    print(
        f"check-voice: {report['n_takes']} takes, {report['hours']:.2f} h, {len(report['flagged'])} flagged, "
        f"{len(report['coverage']['below'])} coverage target(s) below {report['coverage']['min_count']}; "
        f"see {Path(args.work) / 'check_voice.html'}"
    )
    return 0


def add_parser(commands) -> None:
    p = commands.add_parser(
        "check-voice", help="run the score steps on an imported voice set and write check_voice.html/.json (spec 9.3)"
    )
    p.add_argument("--work", type=Path, default=Path("var/dataprep/voice"), help="work dir")
    p.add_argument("--labeller", required=True, help="whisper-hf:<hf id or local dir> or nemo:<hf id or .nemo path>")
    p.add_argument("--workers", type=int, default=1, help="workers for the text and audio-stats steps")
    p.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE, help="phonetic coverage target list")
    p.set_defaults(func=_cmd)
