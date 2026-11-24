"""`select`: apply a filter profile to the step files and write the selection and its report (spec §6.3, §6.4, §7.2, §7.3)."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from app.core.text import bn_normalize

from .profiles import Filter, Flag, Percentile, Profile, Range, load_profile
from .workdir import StepFile, WorkDir


class SelectError(RuntimeError):
    pass


# column -> (step name, step version, params or None to skip the params check).
# Steps from later tasks (dnsmos, brouhaha, speakers, cer_*) are assumed v1.
_AUDIO = ("audio_stats", 1, {})
_TEXT = ("text", 1, {})
COLUMN_STEPS: dict[str, tuple[str, int, dict | None]] = {
    "dur": _AUDIO, "clip_frac": _AUDIO, "silence_share": _AUDIO, "lufs": _AUDIO, "src_sr": _AUDIO,
    "has_latin": _TEXT, "has_digits": _TEXT, "n_chars": _TEXT, "text_clean": _TEXT,
    "dnsmos_p808": ("dnsmos", 1, None), "dnsmos_ovrl": ("dnsmos", 1, None), "dnsmos_sig": ("dnsmos", 1, None), "dnsmos_bak": ("dnsmos", 1, None),
    "snr_db": ("brouhaha", 1, None), "c50_db": ("brouhaha", 1, None), "max_pause_s": ("brouhaha", 1, None),
    "speech_segments": ("brouhaha", 1, None),
    "speaker": ("speakers", 1, None),
}
DERIVED = {"cps": ("n_chars", "dur")}  # cps = n_chars / dur
ALWAYS_STEPS = ("text", "audio_stats")  # text for the transcript and flags, audio_stats for durations

_EXPECTED_PUNCT = set("।?,.!-'\"")
_EXPECTED_EXTRA = {" ", "‌", "‍"}


def _expected(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x0980 <= cp <= 0x09FF
        or (ch.isascii() and ch.isalnum())
        or ch in _EXPECTED_PUNCT
        or ch in _EXPECTED_EXTRA
    )


def cer_step_name(labeller: str) -> str:
    return "cer_" + re.sub(r"[^A-Za-z0-9]", "-", labeller)


@dataclass
class SelectResult:
    rows: pd.DataFrame
    report: dict


def _filter_label(f: Filter) -> str:
    if isinstance(f, Range):
        lo = f"{f.min:g}" if f.min is not None else "-inf"
        hi = f"{f.max:g}" if f.max is not None else "inf"
        return f"{f.column} in [{lo}, {hi}{')' if f.max_exclusive else ']'}"
    if isinstance(f, Percentile):
        return f"{f.column} within p{f.lo:g}-p{f.hi:g}"
    return f"{f.column} == {f.keep}"


def _apply(df: pd.DataFrame, f: Filter) -> pd.DataFrame:
    s = df[f.column]
    if isinstance(f, Range):
        mask = pd.Series(True, index=df.index)
        if f.min is not None:
            mask &= s >= f.min
        if f.max is not None:
            mask &= (s < f.max) if f.max_exclusive else (s <= f.max)
        return df[mask.fillna(False)]
    if isinstance(f, Percentile):
        lo, hi = s.quantile(f.lo / 100), s.quantile(f.hi / 100)
        return df[((s >= lo) & (s <= hi)).fillna(False)]
    return df[(s.astype("boolean") == f.keep).fillna(False)]


def _hours(df: pd.DataFrame) -> float:
    return float(df["dur"].sum(skipna=True)) / 3600.0


def _load(wd: WorkDir, profile: Profile, labeller: str | None) -> tuple[pd.DataFrame, list[str]]:
    needed: dict[str, tuple[str, int, dict | None]] = {s: next(v for v in COLUMN_STEPS.values() if v[0] == s) for s in ALWAYS_STEPS}
    columns = [f.column for f in profile.filters]
    if profile.min_speaker_minutes is not None:
        columns.append("speaker")
    for col in columns:
        if col in DERIVED:
            continue
        if col == "cer":
            if labeller is None:
                raise SelectError(
                    f"profile '{profile.name}' filters on cer but names no labeller; pass --labeller <spec>"
                )
            step = cer_step_name(labeller)
            needed[step] = (step, 1, None)
        elif col in COLUMN_STEPS:
            step, version, params = COLUMN_STEPS[col]
            needed[step] = (step, version, params)
        else:
            raise SelectError(f"profile '{profile.name}' filters on unknown column '{col}'")

    clips = StepFile.read(wd.clips_path, step="import", version=1)
    df = clips[["id", "src_split"]].copy()
    for step, version, params in needed.values():
        path = wd.step_path(step)
        if not path.exists():
            raise SelectError(f"profile '{profile.name}' needs step '{step}' but {path} does not exist; run that step first")
        sf = StepFile.read(path, step=step, version=version, params=params)  # StaleStepError names the step
        sf = sf.rename(columns={"error": f"_error_{step}"})
        sf[f"_has_{step}"] = True
        if f"_error_{step}" not in sf:
            sf[f"_error_{step}"] = None
        df = df.merge(sf, on="id", how="left")
        df[f"_has_{step}"] = df[f"_has_{step}"].fillna(False).astype(bool)
    return df, list(needed)


def _char_audit(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    for rid, text in zip(df["id"], df["text_clean"]):
        for ch in text:
            counts[ch] += 1
            ex = examples.setdefault(ch, [])
            if len(ex) < 5 and rid not in ex:
                ex.append(rid)
    chars = [
        {"char": ch, "codepoint": f"U+{ord(ch):04X}", "count": counts[ch], "examples": examples[ch]}
        for ch in sorted(counts, key=ord)
    ]
    return chars, [c for c in chars if not _expected(c["char"])]


def eligible(
    wd: WorkDir, profile: Profile, labeller: str | None = None, *, skip_columns: frozenset[str] = frozenset()
) -> pd.DataFrame:
    """Rows that pass the profile's filters, ignoring filters on `skip_columns`. Writes nothing."""
    return select(wd, profile, labeller, skip_columns=skip_columns, write=False).rows


def select(
    wd: WorkDir,
    profile: Profile,
    labeller: str | None = None,
    *,
    skip_columns: frozenset[str] = frozenset(),
    write: bool = True,
) -> SelectResult:
    """Apply the profile; `labeller` (from --labeller) overrides `profile.cer_labeller`.

    `skip_columns` drops the filters on those columns (and does not load their steps); used by
    `brouhaha --only-profile` and `listen`. `write=False` skips the selection and report files.
    """
    labeller = labeller or profile.cer_labeller
    if skip_columns:
        profile = profile.model_copy(update={"filters": [f for f in profile.filters if f.column not in skip_columns]})
    df, steps_loaded = _load(wd, profile, labeller)
    df["cps"] = df["n_chars"].astype("float64") / df["dur"].astype("float64")

    report_steps: list[dict] = []

    def run(label: str, fn) -> None:
        nonlocal df
        before = df
        df = fn(df)
        report_steps.append({
            "filter": label, "rows_before": len(before), "rows_after": len(df),
            "hours_before": _hours(before), "hours_after": _hours(df),
        })
        if df.empty:
            raise SelectError(f"profile '{profile.name}': filter '{label}' left no rows")

    def step_errors(d: pd.DataFrame) -> pd.DataFrame:
        bad = d["empty_asr"].fillna(True).astype(bool)
        for step in steps_loaded:
            bad |= ~d[f"_has_{step}"] | d[f"_error_{step}"].notna()
        return d[~bad]

    run("step_errors", step_errors)
    for f in profile.filters:
        run(_filter_label(f), lambda d, f=f: _apply(d, f))
    if profile.drop_latin:
        run("has_latin", lambda d: d[~d["has_latin"].astype(bool)])

    overlap_dropped = 0

    def drop_overlap(d: pd.DataFrame) -> pd.DataFrame:
        nonlocal overlap_dropped
        norm = d["text_clean"].map(lambda t: bn_normalize(t, mode="asr"))
        val_texts = set(norm[d["src_split"] == "val"])
        dup = (d["src_split"] == "train") & norm.isin(val_texts)
        overlap_dropped = int(dup.sum())
        return d[~dup]

    run("train_val_overlap", drop_overlap)

    if profile.min_speaker_minutes is not None:
        minutes = profile.min_speaker_minutes

        def speaker_floor(d: pd.DataFrame) -> pd.DataFrame:
            total = d.groupby("speaker")["dur"].transform("sum")
            return d[total >= minutes * 60]

        run("min_speaker_minutes", speaker_floor)

    chars, unexpected = _char_audit(df)
    report = {
        "profile": profile.name,
        "labeller": labeller,
        "steps": report_steps,
        "chars": chars,
        "unexpected_chars": unexpected,
        "overlap_dropped": overlap_dropped,
    }
    rows = df[[c for c in df.columns if not c.startswith("_")]].reset_index(drop=True)
    if not write:
        return SelectResult(rows=rows, report=report)
    StepFile.write(wd.selection_path(profile.name), rows, step="select", version=1, params={"profile": profile.name})
    report_path = wd.selection_path(profile.name).with_name(f"{profile.name}.report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return SelectResult(rows=rows, report=report)


def format_report(report: dict) -> str:
    lines = [f"{'filter':<34}{'rows before':>12}{'rows after':>12}{'removed':>9}{'hours after':>13}"]
    for s in report["steps"]:
        lines.append(
            f"{s['filter']:<34}{s['rows_before']:>12}{s['rows_after']:>12}"
            f"{s['rows_before'] - s['rows_after']:>9}{s['hours_after']:>13.2f}"
        )
    if report["unexpected_chars"]:
        lines.append("unexpected characters: " + " ".join(f"{c['char']}({c['codepoint']} x{c['count']})" for c in report["unexpected_chars"]))
    return "\n".join(lines)


def _cmd(args: argparse.Namespace) -> int:
    try:
        res = select(WorkDir(args.work), load_profile(args.profile), labeller=args.labeller)
    except SelectError as e:
        print(f"select: {e}")
        return 1
    print(format_report(res.report))
    return 0


def add_parser(commands) -> None:
    p = commands.add_parser("select", help="apply a filter profile and write the selection and report")
    p.add_argument("--work", type=Path, default=Path("var/dataprep/bengaliai"), help="work dir")
    p.add_argument("--profile", required=True, help="profile name or YAML path")
    p.add_argument("--labeller", default=None, help="CER labeller spec; overrides the profile's cer_labeller")
    p.set_defaults(func=_cmd)
