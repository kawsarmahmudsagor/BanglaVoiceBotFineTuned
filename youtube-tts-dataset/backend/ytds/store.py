"""Thread-safe SQLite store for datasets, videos, stages, clips and settings."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from ytds.config import Thresholds

Stage = Literal[
    "download", "cut", "separate", "pieces", "transcribe_a", "transcribe_b", "punctuate_pieces",
    "group", "quality", "select", "punctuate",
    "segment", "transcribe", "fallback",  # legacy: old marks still type-check
]
VideoStatus = Literal["fetching", "awaiting_cut", "queued", "running", "done", "failed", "skipped"]
ClipStatus = Literal[
    "dropped", "auto", "flagged", "needs_review", "pending_llm", "approved", "rejected"
]
TextSource = Literal["stt", "translated"]
DropReason = Literal["length", "dnsmos", "other_speaker", "bad_stt"]

VIDEO_COLS = (
    "url", "title", "channel", "upload_date", "duration_s", "license", "status", "error", "warning",
    "origin", "source", "cuts", "media", "model_a", "model_b",
)
CLIP_COLS = (
    "video_id", "idx", "start_s", "end_s", "dnsmos", "speaker_sim", "drop_reason", "text_a",
    "text_b", "cer", "text_en", "text_translated", "text_final", "text_source", "status", "no_punct",
    "flag", "text_pick", "score_a", "score_b", "range_idx",
)
PIECE_COLS = (
    "range_idx", "start_s", "end_s", "text_a", "text_b", "b_tokens", "b_compression", "b_retry_failed",
    "text", "broken", "no_punct",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets(
  id TEXT PRIMARY KEY, name TEXT UNIQUE, speaker TEXT, created REAL);
CREATE TABLE IF NOT EXISTS videos(
  dataset_id TEXT, id TEXT, url TEXT, title TEXT, channel TEXT, upload_date TEXT,
  duration_s REAL, license TEXT, status TEXT DEFAULT 'queued', error TEXT, warning TEXT,
  origin TEXT, source TEXT, cuts TEXT, media TEXT, model_a TEXT, model_b TEXT,
  PRIMARY KEY(dataset_id, id));
CREATE TABLE IF NOT EXISTS stages(
  dataset_id TEXT, video_id TEXT, stage TEXT, done INTEGER, updated REAL,
  PRIMARY KEY(dataset_id, video_id, stage));
CREATE TABLE IF NOT EXISTS clips(
  dataset_id TEXT, id TEXT, video_id TEXT, idx INTEGER, start_s REAL, end_s REAL,
  dnsmos REAL, speaker_sim REAL, drop_reason TEXT, text_a TEXT, text_b TEXT, cer REAL,
  text_en TEXT, text_translated TEXT, text_final TEXT, text_source TEXT, status TEXT,
  no_punct INTEGER DEFAULT 0,
  flag TEXT, text_pick TEXT, score_a REAL, score_b REAL, range_idx INTEGER,
  PRIMARY KEY(dataset_id, id));
CREATE TABLE IF NOT EXISTS pieces(
  dataset_id TEXT, video_id TEXT, idx INTEGER, range_idx INTEGER, start_s REAL, end_s REAL,
  text_a TEXT, text_b TEXT, b_tokens INTEGER, b_compression REAL, b_retry_failed INTEGER DEFAULT 0,
  text TEXT, broken INTEGER DEFAULT 0, no_punct INTEGER DEFAULT 0,
  PRIMARY KEY(dataset_id, video_id, idx));
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
"""


_NEW_VIDEO_COLS = (
    ("origin", "TEXT"), ("source", "TEXT"), ("cuts", "TEXT"), ("media", "TEXT"),
    ("model_a", "TEXT"), ("model_b", "TEXT"),
)
_NEW_CLIP_COLS = (
    ("flag", "TEXT"), ("text_pick", "TEXT"), ("score_a", "REAL"), ("score_b", "REAL"), ("range_idx", "INTEGER"),
)


def _check(fields: dict[str, Any], allowed: tuple[str, ...]) -> None:
    bad = [k for k in fields if k not in allowed]
    if bad:
        raise ValueError(f"unknown column(s): {bad}")


class Store:
    def __init__(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        with self._lock, self._db:
            self._db.executescript(_SCHEMA)
            self._migrate()

    def _migrate(self) -> None:
        """Bring a database from an older schema up to date. Caller holds the lock and a transaction."""
        for table, cols in (("videos", _NEW_VIDEO_COLS), ("clips", _NEW_CLIP_COLS)):
            have = {r["name"] for r in self._db.execute(f"PRAGMA table_info({table})")}
            for name, decl in cols:
                if name not in have:
                    self._db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
        # stage marks (spec 4.8): a finished old `segment` implies the new segmentation stages, `transcribe` implies select
        now = time.time()
        for old, news in (
            ("segment", ("cut", "pieces", "transcribe_a", "transcribe_b", "punctuate_pieces", "group")),
            ("transcribe", ("select",)),
        ):
            for new in news:
                self._db.execute(
                    "INSERT OR IGNORE INTO stages(dataset_id, video_id, stage, done, updated) "
                    "SELECT dataset_id, video_id, ?, 1, ? FROM stages WHERE stage=? AND done=1",
                    (new, now, old),
                )
        # thresholds: values still at the old defaults move to the new ones
        row = self._db.execute("SELECT value FROM settings WHERE key='thresholds'").fetchone()
        if row:
            try:
                data = json.loads(row["value"])
            except ValueError:
                data = None
            if isinstance(data, dict):
                changed = False
                for key, old, new in (("clip_min_s", 2.0, 15.0), ("clip_max_s", 15.0, 25.0)):
                    if key in data and data[key] == old:
                        data[key] = new
                        changed = True
                if changed:
                    self._db.execute(
                        "UPDATE settings SET value=? WHERE key='thresholds'", (json.dumps(data),)
                    )

    def _write(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock, self._db:
            return self._db.execute(sql, args)

    def _read(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    # datasets
    def create_dataset(self, name: str, speaker: str) -> str:
        ds_id = uuid.uuid4().hex[:12]
        self._write(
            "INSERT INTO datasets(id, name, speaker, created) VALUES (?,?,?,?)",
            (ds_id, name, speaker, time.time()),
        )
        return ds_id

    def get_dataset(self, dataset_id: str) -> dict | None:
        rows = self._read("SELECT * FROM datasets WHERE id=?", (dataset_id,))
        return rows[0] if rows else None

    def list_datasets(self) -> list[dict]:
        return self._read("SELECT * FROM datasets ORDER BY created, id")

    def set_dataset_speaker(self, dataset_id: str, speaker: str) -> None:
        self._write("UPDATE datasets SET speaker=? WHERE id=?", (speaker, dataset_id))

    # videos
    def add_video(
        self, dataset_id: str, video_id: str, url: str, *,
        origin: str = "dashboard", source: str = "youtube", status: VideoStatus = "queued",
    ) -> bool:
        cur = self._write(
            "INSERT OR IGNORE INTO videos(dataset_id, id, url, status, origin, source) VALUES (?,?,?,?,?,?)",
            (dataset_id, video_id, url, status, origin, source),
        )
        return cur.rowcount == 1

    def update_video(self, dataset_id: str, video_id: str, **fields: Any) -> None:
        _check(fields, VIDEO_COLS)
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        self._write(
            f"UPDATE videos SET {sets} WHERE dataset_id=? AND id=?",
            (*fields.values(), dataset_id, video_id),
        )

    def list_videos(self, dataset_id: str) -> list[dict]:
        return self._read("SELECT * FROM videos WHERE dataset_id=? ORDER BY rowid", (dataset_id,))

    def delete_video(self, dataset_id: str, video_id: str) -> None:
        """Remove the video with its stage marks and clips."""
        with self._lock, self._db:
            for table, col in (("videos", "id"), ("stages", "video_id"), ("clips", "video_id"), ("pieces", "video_id")):
                self._db.execute(f"DELETE FROM {table} WHERE dataset_id=? AND {col}=?", (dataset_id, video_id))

    # stages
    def mark_stage(self, dataset_id: str, video_id: str, stage: Stage, done: bool) -> None:
        self._write(
            "INSERT OR REPLACE INTO stages(dataset_id, video_id, stage, done, updated) "
            "VALUES (?,?,?,?,?)",
            (dataset_id, video_id, stage, int(done), time.time()),
        )

    def stage_done(self, dataset_id: str, video_id: str, stage: Stage) -> bool:
        rows = self._read(
            "SELECT done FROM stages WHERE dataset_id=? AND video_id=? AND stage=?",
            (dataset_id, video_id, stage),
        )
        return bool(rows and rows[0]["done"])

    # clips
    def replace_clips(self, dataset_id: str, video_id: str, clips: list[dict]) -> None:
        for c in clips:
            _check({k: v for k, v in c.items() if k != "id"}, CLIP_COLS)
        with self._lock, self._db:
            self._db.execute(
                "DELETE FROM clips WHERE dataset_id=? AND video_id=?", (dataset_id, video_id)
            )
            for c in clips:
                row = {**c, "dataset_id": dataset_id, "video_id": video_id}
                cols = ", ".join(row)
                marks = ", ".join("?" for _ in row)
                self._db.execute(
                    f"INSERT INTO clips({cols}) VALUES ({marks})", tuple(row.values())
                )

    def update_clip(self, dataset_id: str, clip_id: str, **fields: Any) -> None:
        _check(fields, CLIP_COLS)
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        self._write(
            f"UPDATE clips SET {sets} WHERE dataset_id=? AND id=?",
            (*fields.values(), dataset_id, clip_id),
        )

    def update_clip_if(self, dataset_id: str, clip_id: str, expect: dict[str, Any], **fields: Any) -> bool:
        """Compare-and-set: apply `fields` only while every `expect` column still holds its value (NULL-safe).
        Returns False, writing nothing, when the clip changed since it was read (for example a review edit)."""
        _check(fields, CLIP_COLS)
        _check(expect, CLIP_COLS)
        if not fields:
            return False
        sets = ", ".join(f"{k}=?" for k in fields)
        conds = "".join(f" AND {k} IS ?" for k in expect)
        cur = self._write(
            f"UPDATE clips SET {sets} WHERE dataset_id=? AND id=?{conds}",
            (*fields.values(), dataset_id, clip_id, *expect.values()),
        )
        return cur.rowcount == 1

    def get_clip(self, dataset_id: str, clip_id: str) -> dict | None:
        rows = self._read("SELECT * FROM clips WHERE dataset_id=? AND id=?", (dataset_id, clip_id))
        return rows[0] if rows else None

    def clips(
        self,
        dataset_id: str,
        *,
        video_id: str | None = None,
        statuses: list[str] | None = None,
    ) -> list[dict]:
        sql = "SELECT * FROM clips WHERE dataset_id=?"
        args: list[Any] = [dataset_id]
        if video_id is not None:
            sql += " AND video_id=?"
            args.append(video_id)
        if statuses is not None:
            if not statuses:
                return []
            sql += f" AND status IN ({', '.join('?' for _ in statuses)})"
            args.extend(statuses)
        return self._read(sql + " ORDER BY video_id, idx", tuple(args))

    # pieces
    def replace_pieces(self, dataset_id: str, video_id: str, pieces: list[dict]) -> None:
        for p in pieces:
            _check({k: v for k, v in p.items() if k != "idx"}, PIECE_COLS)
        with self._lock, self._db:
            self._db.execute("DELETE FROM pieces WHERE dataset_id=? AND video_id=?", (dataset_id, video_id))
            for p in pieces:
                row = {**p, "dataset_id": dataset_id, "video_id": video_id}
                cols = ", ".join(row)
                marks = ", ".join("?" for _ in row)
                self._db.execute(f"INSERT INTO pieces({cols}) VALUES ({marks})", tuple(row.values()))

    def pieces(self, dataset_id: str, video_id: str) -> list[dict]:
        return self._read(
            "SELECT * FROM pieces WHERE dataset_id=? AND video_id=? ORDER BY idx", (dataset_id, video_id)
        )

    def update_piece(self, dataset_id: str, video_id: str, idx: int, **fields: Any) -> None:
        _check(fields, PIECE_COLS)
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        self._write(
            f"UPDATE pieces SET {sets} WHERE dataset_id=? AND video_id=? AND idx=?",
            (*fields.values(), dataset_id, video_id, idx),
        )

    # settings
    def get_setting(self, key: str, default: str | None = None) -> str | None:
        rows = self._read("SELECT value FROM settings WHERE key=?", (key,))
        return rows[0]["value"] if rows else default

    def set_setting(self, key: str, value: str) -> None:
        self._write("INSERT OR REPLACE INTO settings(key, value) VALUES (?,?)", (key, value))

    def get_thresholds(self) -> Thresholds:
        raw = self.get_setting("thresholds")
        return Thresholds.model_validate_json(raw) if raw else Thresholds()

    def set_thresholds(self, t: Thresholds) -> None:
        self.set_setting("thresholds", t.model_dump_json())
