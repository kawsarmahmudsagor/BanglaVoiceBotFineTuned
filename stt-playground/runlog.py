"""Append-only JSONL log of everything the playground does: model loads, file transcriptions and live turns.

One JSON object per line, all durations in seconds. Default path: stt-playground/logs/playground.jsonl
(override with STT_LOG). Read it with e.g. `pandas.read_json(path, lines=True)` or `jq`.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(os.environ.get("STT_LOG", Path(__file__).resolve().parent / "logs" / "playground.jsonl"))
SESSION = f"{socket.gethostname()}-{os.getpid()}-{int(time.time())}"
_lock = threading.Lock()
_gpu: str | None = None


def gpu_name() -> str:
    global _gpu
    if _gpu is None:
        try:
            import torch

            _gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            _gpu = "unknown"
    return _gpu


def log(event: str, **fields) -> None:
    """Write one record. Never raises: a logging failure must not break transcription."""
    rec = {"ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds"), "event": event,
           "session": SESSION, "gpu": gpu_name(), **fields}
    try:
        with _lock:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"runlog: cannot write {LOG_PATH}: {e}")
