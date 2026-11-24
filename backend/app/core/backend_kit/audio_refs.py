"""Audio-reference helpers shared by the audio converters (whisper, ctc): a threaded decodability probe,
progress reporting and the stale-path check. Light: soundfile and the audio helpers load inside functions."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

SAMPLE_RATE = 16000
PROBE_THREADS = 16
PROGRESS_EVERY = 10_000
CHUNK = 1000


def probe_duration(path, sr: int = SAMPLE_RATE) -> float:
    """Seconds of audio, verifying the file is decodable: a header read, else a full decode.
    Raises if the file cannot be decoded."""
    import soundfile as sf

    try:
        return float(sf.info(str(path)).duration)
    except Exception:  # noqa: BLE001 - header unreadable for this format: decode instead
        from app.core.backend_kit.audio import load_audio

        return len(load_audio(path, sr)) / sr


def check_audio_refs(paths: list[str], cache: str = "") -> None:
    """Raise if a prepared dataset's audio files are gone (the cache outlived its source files)."""
    for p in paths:
        if not Path(p).exists():
            raise RuntimeError(f"prepared cache {cache} points at missing audio {p}; "
                               "delete the cache and re-run prepare")


def _attempt(probe: Callable):
    def run(path):
        try:
            return probe(path)
        except Exception as e:  # noqa: BLE001 - reported as an undecodable skip by the caller
            return e

    return run


def probed_rows(items: list[dict], probe: Callable,
                tick: Callable[[int], None] = lambda n: None) -> Iterator[tuple[dict, object]]:
    """Yield (row, probe result or the Exception it raised) in row order. Files are probed by a thread
    pool a chunk at a time; `tick(n)` is called after each chunk's rows have been consumed."""
    from concurrent.futures import ThreadPoolExecutor

    run = _attempt(probe)
    with ThreadPoolExecutor(PROBE_THREADS) as pool:
        for lo in range(0, len(items), CHUNK):
            chunk = items[lo:lo + CHUNK]
            yield from zip(chunk, pool.map(run, [r["audio_path"] for r in chunk]))
            tick(len(chunk))


def progress_ticker(rep, total: int) -> Callable[[int], None]:
    """tick(n) reporting `prepared done/total rows` every PROGRESS_EVERY rows and at the end."""
    from app.core.contract import Phase

    state = {"done": 0, "reported": 0}

    def tick(n: int) -> None:
        state["done"] += n
        if rep is not None and (state["done"] - state["reported"] >= PROGRESS_EVERY or state["done"] == total):
            state["reported"] = state["done"]
            rep.status(Phase.prepare, state["done"], total, f"prepared {state['done']}/{total} rows")

    return tick
