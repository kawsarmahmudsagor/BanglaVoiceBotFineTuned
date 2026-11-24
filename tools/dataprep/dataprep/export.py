"""`export`: write the saved selection in the platform's upload layout (spec §4.4, §8.3).

asr: metadata.csv (file,text,split), audio/<id>.flac (16 kHz mono PCM_16), curation.json.
tts: metadata.csv (file,text,speaker,split), wavs/<id>.wav (24 kHz PCM_16, trimmed, -23 LUFS), curation.json.
Every file is written as `<name>.tmp-<pid>` then `os.replace`d, never rewritten in place, because
hard-linked dataset versions share inodes with earlier exports.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .pool import run_rows
from .profiles import Profile, load_profile
from .workdir import StepFile, WorkDir

TARGET_SR = 16000
TTS_SR = 24000

# Ingest form fields (backend Provenance). Bengali.AI's license is unchecked, so never commercial.
BENGALIAI_PROVENANCE = {
    "source": "Bengali.AI Speech Recognition (Kaggle competition corpus)",
    "license": "unverified (Kaggle competition terms; not checked)",
    "consent": "unverified",
    "commercial_ok": False,
    "notes": "Curated with tools/dataprep; license and speaker consent not yet checked.",
}


class ExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class Layout:
    subdir: str
    suffix: str
    columns: tuple[str, ...]


ASR_LAYOUT = Layout("audio", ".flac", ("file", "text", "split"))
TTS_LAYOUT = Layout("wavs", ".wav", ("file", "text", "speaker", "split"))
_TMP_SUFFIX = re.compile(r"\.tmp-\d+$")


def _tmp(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp-{os.getpid()}")


def _transcode(row: dict) -> dict:
    import librosa
    import soundfile as sf
    import soxr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(row["src_path"], sr=None, mono=True)
    if sr != TARGET_SR:
        y = soxr.resample(y, sr, TARGET_SR)
    sf.write(row["tmp"], y, TARGET_SR, format="FLAC", subtype="PCM_16")
    return {"id": row["id"]}


def _transcode_tts(row: dict) -> dict:
    """Decode the original file (never ASR audio), process per spec 8.3, write 24 kHz PCM_16 WAV."""
    import librosa
    import soundfile as sf

    from .audio_proc import process_tts

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(row["src_path"], sr=None, mono=True)
    out = process_tts(y, sr, row["segments"], out_sr=TTS_SR)
    sf.write(row["tmp"], out, TTS_SR, format="WAV", subtype="PCM_16")
    return {"id": row["id"], "band_limited": bool(sr < TTS_SR)}


def _tool_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _step_versions(wd: WorkDir) -> dict[str, int]:
    versions: dict[str, int] = {}
    for p in sorted(wd.root.glob("*.parquet")):
        meta = StepFile._meta(p)
        versions[meta["step"]] = meta["version"]
    return versions


def _write_atomic(path: Path, write) -> None:
    tmp = _tmp(path)
    write(tmp)
    os.replace(tmp, path)


def _tts_inputs(wd: WorkDir, ids: list[str], speaker: str | None) -> tuple[dict[str, list], pd.Series | str]:
    """Per-id speech segments and the speaker source. Raises before anything is written."""
    bro_path = wd.step_path("brouhaha")
    if not bro_path.exists():
        raise ExportError(f"TTS export needs speech segments but {bro_path} is missing; run `brouhaha` first")
    bro = StepFile.read(bro_path, step="brouhaha", version=1).set_index("id")["speech_segments"]
    bad = [i for i in ids if i not in bro.index or bro[i] is None or len(bro[i]) == 0]
    if bad:
        raise ExportError(f"{len(bad)} selected id(s) have no brouhaha speech_segments: {bad[:10]}")
    segments = {i: [[float(a), float(b)] for a, b in bro[i]] for i in ids}
    if speaker is not None:
        return segments, speaker
    spk_path = wd.step_path("speakers")
    if not spk_path.exists():
        raise ExportError(f"no speaker labels: {spk_path} is missing; run `speakers` or pass --speaker <name>")
    spk = StepFile.read(spk_path, step="speakers", version=1).set_index("id")["speaker"]
    bad = [i for i in ids if i not in spk.index or pd.isna(spk[i])]
    if bad:
        raise ExportError(f"{len(bad)} selected id(s) have no speaker in speakers.parquet (or pass --speaker): {bad[:10]}")
    return segments, spk


def export(
    wd: WorkDir, profile: Profile, dataset: str, *, workers: int = 1, out: Path | None = None,
    speaker: str | None = None,
) -> Path:
    sel_path = wd.selection_path(profile.name)
    report_path = sel_path.with_name(f"{profile.name}.report.json")
    if not sel_path.exists() or not report_path.exists():
        raise ExportError(f"no saved selection for profile '{profile.name}' ({sel_path}); run `select` first")
    sel = StepFile.read(sel_path, step="select", version=1, params={"profile": profile.name})
    if sel.empty:
        raise ExportError(f"selection for profile '{profile.name}' is empty; nothing to export")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    clips = StepFile.read(wd.clips_path, step="import", version=1)
    src = clips.set_index("id")["src_path"]
    missing = [i for i in sel["id"] if i not in src.index]
    if missing:
        raise ExportError(f"selected ids not in clips.parquet (stale selection? rerun `select`): {missing[:10]}")

    tts = profile.task == "tts"
    layout = TTS_LAYOUT if tts else ASR_LAYOUT
    ids = list(sel["id"])
    segments, speakers = _tts_inputs(wd, ids, speaker) if tts else ({}, None)

    dest = Path(out) if out is not None else wd.export_dir(dataset)
    audio = dest / layout.subdir
    audio.mkdir(parents=True, exist_ok=True)
    jobs = [{"id": i, "src_path": str(src[i]), "tmp": str(_tmp(audio / f"{i}{layout.suffix}")),
             "segments": segments.get(i)} for i in ids]
    results = run_rows(_transcode_tts if tts else _transcode, jobs, workers=workers, desc="export audio")
    failed = [r for r in results if r["error"]]
    if failed:
        for j in jobs:
            Path(j["tmp"]).unlink(missing_ok=True)
        detail = "; ".join(f"{r['id']} ({r['error']})" for r in failed[:20])
        raise ExportError(f"{len(failed)} clip(s) failed to process, export aborted: {detail}")

    for i in ids:
        os.replace(_tmp(audio / f"{i}{layout.suffix}"), audio / f"{i}{layout.suffix}")

    def write_meta(tmp: Path) -> None:
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(layout.columns)
            for rid, text, split in zip(sel["id"], sel["text_clean"], sel["src_split"]):
                row = [f"{layout.subdir}/{rid}{layout.suffix}", text]
                if tts:
                    row.append(speakers if isinstance(speakers, str) else speakers[rid])
                w.writerow([*row, "" if pd.isna(split) else split])

    # Metadata first, so it never points at audio the sweep below is about to delete.
    _write_atomic(dest / "metadata.csv", write_meta)
    keep = {f"{i}{layout.suffix}" for i in ids}
    for p in audio.iterdir():
        stale = p.name not in keep and (p.suffix == layout.suffix or _TMP_SUFFIX.search(p.name))
        if stale and p.is_file() and not p.is_symlink():
            p.unlink()

    curation = {
        "source": wd.root.name,
        "profile": {"name": profile.name, **profile.model_dump(mode="json", exclude={"name"})},
        "step_versions": _step_versions(wd),
        "report": report,
        "tool_commit": _tool_commit(),
        "provenance": BENGALIAI_PROVENANCE,
    }
    if tts:
        curation["band_limited"] = sum(1 for r in results if r["band_limited"])
    _write_atomic(
        dest / "curation.json",
        lambda tmp: tmp.write_text(json.dumps(curation, ensure_ascii=False, indent=2), encoding="utf-8"),
    )
    return dest


def _cmd(args: argparse.Namespace) -> int:
    wd = WorkDir(args.work)
    try:
        profile = load_profile(args.profile)
        dest = export(wd, profile, args.dataset, workers=args.workers, out=args.out, speaker=args.speaker)
    except (ExportError, FileNotFoundError) as e:
        print(f"export: {e}")
        return 1
    print(f"exported to {dest}")
    return 0


def add_parser(commands) -> None:
    p = commands.add_parser(
        "export",
        help="write the saved selection as a dataset: asr (audio/*.flac) or tts (wavs/*.wav), per the profile task",
        description=(
            "Export the selection written by `select`. For local-path ingest (platform Task 7) put the export "
            "dir under the platform's var/import/, on the same filesystem as its dataset store so ingest can hard-link."
        ),
    )
    p.add_argument("--work", type=Path, default=Path("var/dataprep/bengaliai"), help="work dir")
    p.add_argument("--profile", required=True, help="profile name or YAML path")
    p.add_argument("--dataset", required=True, help="dataset name (export dir name under the work dir)")
    p.add_argument("--out", type=Path, default=None,
                   help="export dir (default <work>/export/<dataset>); use a dir under the platform's var/import/")
    p.add_argument("--speaker", default=None,
                   help="single speaker name for every clip (tts, e.g. the voice profile); "
                        "default: pseudo-speakers from speakers.parquet")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="parallel transcode workers")
    p.set_defaults(func=_cmd)
