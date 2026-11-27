"""GPU worker: keeps one model on the GPU at a time, plus an OOM retry helper."""
from __future__ import annotations

import gc
import threading
from typing import Any, Callable, TypeVar

import torch

T = TypeVar("T")


def torch_empty_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class GpuWorker:
    def __init__(self, loaders: dict[str, Callable[[], Any]], empty_cache: Callable[[], None] = torch_empty_cache):
        self._loaders = loaders
        self._empty_cache = empty_cache
        self._lock = threading.Lock()
        self._model: Any = None
        self.loaded: str | None = None

    def use(self, name: str) -> Any:
        with self._lock:
            if self.loaded == name:
                return self._model
            loader = self._loaders[name]
            self._drop()
            self._model = loader()
            self.loaded = name
            return self._model

    def release(self) -> None:
        with self._lock:
            self._drop()

    def unload(self, name: str) -> None:
        """Drop the model only if `name` is the one loaded (a Settings change of that model)."""
        with self._lock:
            if self.loaded == name:
                self._drop()

    def _drop(self) -> None:
        if self.loaded is None:
            return
        self._model = None
        self.loaded = None
        self._empty_cache()


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def run_with_oom_retry(
    fn: Callable[[float], T], first: float, empty_cache: Callable[[], None] = torch_empty_cache
) -> T:
    try:
        return fn(first)
    except Exception as exc:
        if not _is_oom(exc):
            raise
    empty_cache()
    return fn(first / 2)
