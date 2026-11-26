"""Export a dataset in the platform `tts` layout: wavs/, metadata.csv, curation.json (spec §7)."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import soundfile as sf

from .runner import vocals_path
from .splits import assign_splits

EXPORT_STATUSES = ("auto", "approved")
PAD_S = 0.15
OUT_SR = 24000
HEADER = ["file", "text", "speaker", "split"]
_FILE_ERR_CODES = {"undecodable", "missing_transcript", "missing_audio", "duplicate_file"}
_DROP_WARNINGS = {"duration_out_of_range", "loudness_out_of_range"}


class NothingToExport(Exception):
    """No clip could be rendered into the export; the previous export (if any) is left untouched."""


class UnreviewedTranslated(Exception):
    def __init__(self, count: int):
        super().__init__(f"{count} translated clip(s) have not been reviewed")
        self.count = count


def _select(store, dataset_id: str) -> tuple[list[dict], int, int]:
    """(exportable clips, unreviewed translated clips, clips with empty text). Unreviewed translated clips
    (needs_review + translated) are counted and never exported."""
    clips = store.clips(dataset_id, statuses=[*EXPORT_STATUSES, "needs_review"])
    kept: list[dict] = []
    unreviewed = empty = 0
    for c in clips:
        is_unrev = c["status"] == "needs_review"
        if is_unrev and c["text_source"] != "translated":
            continue
        if not (c["text_final"] or "").strip():
            empty += 1
            continue
        if is_unrev:
            unreviewed += 1
            continue
        kept.append(c)
    return kept, unreviewed, empty


def exportable(store, dataset_id: str) -> tuple[list[dict], int]:
    """(clips to export, number of unreviewed translated clips). Flagged-but-unapproved and unreviewed
    translated clips are excluded."""
    kept, unreviewed, _ = _select(store, dataset_id)
    return kept, unreviewed


def _git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                           capture_output=True, text=True, timeout=10)
        return (r.stdout.strip() or None) if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _model_ids() -> dict:
    from .pipeline.fallback import TRANSLATE_MODEL
    from .pipeline.transcribe import MODEL_A, MODEL_B
    return {"model_a": MODEL_A, "model_b": MODEL_B, "fallback": TRANSLATE_MODEL,
            "separation": "htdemucs", "vad": "silero-vad"}


def _write_metadata(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _render(dataset_id: str, clip: dict, data_dir: Path):
    from dataprep.audio_proc import process_tts

    path = vocals_path(data_dir, dataset_id, clip["video_id"])
    info = sf.info(path)
    sr = info.samplerate
    s0 = max(0.0, clip["start_s"] - PAD_S)
    e0 = min(info.frames / sr, clip["end_s"] + PAD_S)
    y, _ = sf.read(path, start=int(round(s0 * sr)), stop=int(round(e0 * sr)), dtype="float32")
    return process_tts(y, sr, [(clip["start_s"] - s0, clip["end_s"] - s0)], pad_s=PAD_S, out_sr=OUT_SR)


def export_dataset(
    store, dataset_id: str, out_dir: Path, on_progress: Callable[[int, int], None], *,
    data_dir: Path, skip_unreviewed_translated: bool = False,
) -> dict:
    from app.datasets.validate_audio import validate_audio

    out_dir = Path(out_dir)
    clips, unreviewed, empty = _select(store, dataset_id)
    if unreviewed and not skip_unreviewed_translated:
        raise UnreviewedTranslated(unreviewed)
    ds = store.get_dataset(dataset_id)
    speaker = ds.get("speaker") or ds["name"]
    splits = assign_splits(clips)

    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "wavs").mkdir(parents=True)

    rows: list[dict] = []
    seconds: dict[str, float] = {}
    process_errors = 0
    total = len(clips)
    try:
        for i, c in enumerate(clips, 1):
            try:
                audio = _render(dataset_id, c, Path(data_dir))
            except ValueError:
                process_errors += 1
            else:
                rel = f"wavs/{c['id']}.wav"
                sf.write(tmp / rel, audio, OUT_SR, subtype="PCM_16")
                seconds[c["id"]] = len(audio) / OUT_SR
                rows.append({"file": rel, "text": c["text_final"].strip(), "speaker": speaker, "split": splits[c["id"]]})
            on_progress(i, total)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    _write_metadata(tmp / "metadata.csv", rows)

    if not rows:
        shutil.rmtree(tmp, ignore_errors=True)
        raise NothingToExport(f"no clip could be rendered ({process_errors} failed)")
    report, valid_rows = validate_audio(tmp, "tts")
    whole = [e for e in report.errors if e.code not in _FILE_ERR_CODES]
    if whole:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"export validation failed: {whole[0].code}: {whole[0].message}")
    valid_files = {r["file"] for r in valid_rows}
    file_of = {r["sample_id"]: r["file"] for r in valid_rows}
    bad = {r["file"] for r in rows} - valid_files
    bad |= {file_of[w.sample_id] for w in report.warnings
            if w.code in _DROP_WARNINGS and w.sample_id in file_of}
    if bad:
        for rel in bad:
            (tmp / rel).unlink(missing_ok=True)
        rows = [r for r in rows if r["file"] not in bad]
        if not rows:
            shutil.rmtree(tmp, ignore_errors=True)
            raise NothingToExport("every clip failed export validation")
        _write_metadata(tmp / "metadata.csv", rows)
    by_file = {f"wavs/{c['id']}.wav": c for c in clips}
    final = [by_file[r["file"]] for r in rows]

    hours = {s: 0.0 for s in ("train", "val", "test")}
    for r in rows:
        hours[r["split"]] += seconds[by_file[r["file"]]["id"]] / 3600
    dropped_s: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for c in store.clips(dataset_id, statuses=["dropped", "rejected"]):
        dropped_s[c["video_id"]][c["drop_reason"] or c["status"]] += c["end_s"] - c["start_s"]
    videos = []
    for v in store.list_videos(dataset_id):
        mine = [c for c in final if c["video_id"] == v["id"]]
        videos.append({
            "video_id": v["id"], "url": v["url"], "title": v["title"], "channel": v["channel"],
            "upload_date": v["upload_date"], "license": v["license"], "clips_kept": len(mine),
            "seconds_kept": round(sum(seconds[c["id"]] for c in mine), 3),
            "seconds_dropped": {k: round(s, 3) for k, s in dropped_s[v["id"]].items()},
        })
    curation = {
        "tool": "youtube-tts-dataset", "git_commit": _git_commit(),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"id": dataset_id, "name": ds["name"], "speaker": speaker},
        "thresholds": store.get_thresholds().model_dump(), "models": _model_ids(),
        "videos": videos,
        "counts": {
            "total_clips": len(final),
            "hours": {k: round(h, 4) for k, h in hours.items()},
            "text_source": dict(Counter(c["text_source"] for c in final)),
            "no_punct": sum(1 for c in final if c["no_punct"]),
            "unreviewed_translated_skipped": unreviewed,
            "empty_text_excluded": empty,
            "process_error": process_errors,
        },
        "validation_dropped": len(bad),
        "commercial_use": False,
    }
    (tmp / "curation.json").write_text(json.dumps(curation, ensure_ascii=False, indent=2), encoding="utf-8")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    return {"out_dir": str(out_dir), "clips": len(final), "hours": curation["counts"]["hours"],
            "process_error": process_errors, "empty_text_excluded": empty,
            "validation_dropped": len(bad), "unreviewed_translated_skipped": unreviewed}
