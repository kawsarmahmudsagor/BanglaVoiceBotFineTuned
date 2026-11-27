"""Streaming turn decode for the live tab: transcribe chunks while the speaker talks, flush the rest at turn end.

RealtimeSTT's design decodes the whole turn after the turn-end silence, so STT latency grows with the turn length.
Here the recorder still owns capture, VAD and turn end (so the endpointing is identical), but while a turn is being
recorded a TurnStreamer thread watches the recorder's frame buffer and commits a chunk whenever:

1. the speaker paused for at least `pause_s` (shorter than the turn-end silence) after >= MIN_CHUNK_S of audio: the
   chunk ends in that pause; or
2. WINDOW_S of audio passed without such a pause: the chunk ends at the quietest 20 ms in the last 2 s, and the next
   chunk starts OVERLAP_S earlier (words both decoded are removed with whisper_chunking.join_overlap).

Each chunk is decoded by the recorder's own engine (`perform_final_transcription(audio_bytes=...)`, same model and
child process, so no second model copy). At turn end only the uncommitted rest is decoded, after waiting for a
chunk decode still in flight; a rest with no speech is skipped. Chunks are decoded without the text before them, so
words at a chunk edge can come out slightly differently than in a whole-turn decode.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from whisper_chunking import OVERLAP_S, WINDOW_S, _quietest_split, join_overlap

SR = 16000
MIN_CHUNK_S = 1.5  # no chunk shorter than this: very short clips decode badly
POLL_S = 0.1
VAD_PAD_MS = 100


def _regions(audio: np.ndarray, pause_s: float) -> list[tuple[int, int]]:
    """Silero speech regions; a region only ends after `pause_s` of silence, so trailing speech runs to the end."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(min_silence_duration_ms=int(pause_s * 1000), speech_pad_ms=VAD_PAD_MS)
    return [(t["start"], t["end"]) for t in get_speech_timestamps(audio, opts)]


def _frames_audio(rec) -> np.ndarray:
    from RealtimeSTT.core.recording_buffers import snapshot_frames

    raw = b"".join(snapshot_frames(rec))
    return np.frombuffer(raw[: len(raw) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0


class TurnStreamer:
    """One per recorded turn. Started from on_recording_start; finish() is called once the recorder has the turn."""

    def __init__(self, rec, pause_s: float, is_current):
        self.rec, self.pause_s, self.is_current = rec, pause_s, is_current
        self.committed = 0  # samples of the turn already decoded (or in flight)
        self.overlap_next = False
        self.chunks: list[dict] = []
        self.started = time.perf_counter()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.wait(POLL_S):
            if not self.is_current() or not self.rec.is_recording:
                continue
            try:
                audio = _frames_audio(self.rec)
                cut = self._find_cut(audio)
                if cut:
                    self._decode(audio, *cut)
            except Exception as e:  # noqa: BLE001 - a failed chunk is decoded again in the flush
                print(f"stream chunk failed: {e}")

    def _find_cut(self, audio: np.ndarray) -> tuple[int, str] | None:
        """(end sample of the next chunk, reason) or None."""
        u = audio[self.committed:]
        if len(u) < MIN_CHUNK_S * SR:
            return None
        regions = _regions(u, self.pause_s)
        if not regions:
            return None
        cuts = [(a_end + b_start) // 2 for (_, a_end), (b_start, _) in zip(regions, regions[1:])]
        if len(u) - regions[-1][1] > VAD_PAD_MS * SR // 1000:  # last region ended: speaker is in a pause now
            cuts.append(regions[-1][1])
        cuts = [c for c in cuts if c >= MIN_CHUNK_S * SR]
        if cuts:
            return self.committed + cuts[-1], "pause"
        win, search = int(WINDOW_S * SR), 2 * SR
        if len(u) > win:
            return self.committed + _quietest_split(u, win - search, win), "forced"
        return None

    def _decode(self, audio: np.ndarray, end: int, reason: str, final: bool = False):
        start = self.committed
        clip = np.ascontiguousarray(audio[start:end], dtype=np.float32)
        overlap = self.overlap_next
        if reason == "forced":
            self.committed, self.overlap_next = max(end - int(OVERLAP_S * SR), start + 1), True
        else:
            self.committed, self.overlap_next = end, False
        t0 = time.perf_counter()
        text = (self.rec.perform_final_transcription(audio_bytes=clip) or "").strip()
        self.chunks.append({"start_s": round(start / SR, 2), "end_s": round(end / SR, 2), "reason": reason,
                            "overlap_prev": overlap, "decoded_at_s": round(t0 - self.started, 2),
                            "decode_s": round(time.perf_counter() - t0, 4), "text": text})

    def cancel(self):
        self._stop.set()

    def finish(self, turn_audio: np.ndarray) -> tuple[str, dict]:
        """Stop chunking, decode the uncommitted rest of `turn_audio`, return (text, log fields)."""
        t0 = time.perf_counter()
        self._stop.set()
        self._thread.join()  # waits for a chunk decode in flight
        t1 = time.perf_counter()
        audio = np.asarray(turn_audio, dtype=np.float32)
        rest = audio[self.committed:]
        streamed = len(self.chunks)
        if len(rest) and _regions(rest, self.pause_s):
            self._decode(audio, len(audio), "flush")
        t2 = time.perf_counter()
        return self._join(), {
            "streaming": True, "chunk_pause_s": self.pause_s, "chunks": self.chunks,
            "chunks_while_speaking": streamed, "streamed_audio_s": round(min(self.committed, len(audio)) / SR, 3)
            if streamed else 0.0, "flush_wait_s": round(t1 - t0, 4), "flush_decode_s": round(t2 - t1, 4),
            "rest_audio_s": round(max(len(rest), 0) / SR, 3), "rest_decoded": len(self.chunks) > streamed}

    def _join(self) -> str:
        text = ""
        for i, c in enumerate(self.chunks):
            t = c["text"]
            if i < len(self.chunks) - 1 and t.endswith(".") and not t.endswith(".."):
                t = t[:-1]  # the recorder's ensure_sentence_ends_with_period, wrong mid-turn
            if c["overlap_prev"] and text:
                t = join_overlap(text, t)
            text = f"{text} {t}".strip()
        return text
