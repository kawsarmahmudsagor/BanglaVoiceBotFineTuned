"""Thread-safe event bus: worker threads publish, asyncio subscribers consume."""
from __future__ import annotations

import asyncio
import threading
import time


class EventBus:
    def __init__(self, min_interval_s: float = 0.25, clock=time.monotonic):
        self._min_interval_s = min_interval_s
        self._clock = clock
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()
        self._last_progress: dict[tuple, float] = {}
        # Only touched on the event loop thread.
        self._subs: dict[str, list[asyncio.Queue]] = {}

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self, dataset_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subs.setdefault(dataset_id, []).append(q)
        return q

    def unsubscribe(self, dataset_id: str, queue: asyncio.Queue) -> None:
        queues = self._subs.get(dataset_id, [])
        if queue in queues:
            queues.remove(queue)
        if not queues:
            self._subs.pop(dataset_id, None)

    def publish(self, event: dict) -> None:
        """Thread-safe. Events published before bind_loop() are dropped."""
        loop = self._loop
        if loop is None or not self._admit(event):
            return
        loop.call_soon_threadsafe(self._deliver, event)

    def _admit(self, event: dict) -> bool:
        if event.get("type") != "progress":
            return True
        if event.get("done") == event.get("total"):
            return True
        key = (event.get("dataset_id"), event.get("video_id"), event.get("stage"))
        now = self._clock()
        with self._lock:
            last = self._last_progress.get(key)
            if last is not None and now - last < self._min_interval_s:
                return False
            self._last_progress[key] = now
            return True

    def _deliver(self, event: dict) -> None:
        for q in list(self._subs.get(event.get("dataset_id"), [])):
            q.put_nowait(event)
