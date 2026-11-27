"""Bangla-safe windowing for faster-whisper's batched pipeline (the RealtimeSTT recorder's path and the file tab's).

Stock behaviour with VAD on: speech is merged into windows of up to 30 s, each decoded once. Two failures on Bangla:
the 224-token cap (fixed separately via max_length) and the fine-tunes ending the transcript at a pause inside a long
window. This module replaces the windowing, still as one batched decode call:

1. Windows of at most WINDOW_S, built from VAD speech regions, split at pauses (and always at pauses > GAP_SPLIT_S).
2. A single speech region longer than WINDOW_S (unbroken speech) is split at its quietest 20 ms frame in the last
   2 s before the limit, and the next window starts OVERLAP_S earlier. The words both windows decoded are removed
   from the second when the texts are joined.
3. A window whose output looks cut is decoded again as RETRY_WINDOW_S sub-windows: it hit the token cap, or it
   produced fewer than MIN_TOK_PER_S tokens per second of window (complete Bangla windows measured 23-31; windows
   that lost sentences 8-10). The retry is kept only if it
   yields more tokens without a repetition loop (compression ratio <= 2.4).

Normal turns pay one extra VAD pass (milliseconds); only suspicious windows pay a second decode.
Every call writes one `whisper_windows` record to the playground log with the windows, splits and retries.
"""
from __future__ import annotations

import dataclasses
import difflib
import os
import re

import numpy as np

SR = 16000
WINDOW_S = float(os.environ.get("WHISPER_WINDOW_S", "10"))
RETRY_WINDOW_S = float(os.environ.get("WHISPER_RETRY_WINDOW_S", "5"))
OVERLAP_S = float(os.environ.get("WHISPER_OVERLAP_S", "1.0"))
MIN_TOK_PER_S = float(os.environ.get("WHISPER_MIN_TOK_PER_S", "12"))
# A pause this long always ends a window (a sentence boundary; long silence invites early EOT). Measured between
# VAD regions, which are padded by 400 ms each side, so 0.2 s here is a ~1 s pause in the audio.
GAP_SPLIT_S = 0.2
MIN_RETRY_AUDIO_S = 2.5  # shorter windows can legitimately hold one or two words
MAX_COMPRESSION = 2.4  # faster-whisper's own repetition threshold


@dataclasses.dataclass
class Window:
    start: int  # samples
    end: int
    overlap_prev: bool = False  # starts OVERLAP_S before the previous window's end (forced split)


def _speech_regions(audio: np.ndarray) -> list[tuple[int, int]]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(min_silence_duration_ms=160)  # batched pipeline's value; speech_pad_ms stays 400
    return [(t["start"], t["end"]) for t in get_speech_timestamps(audio, opts)]


def _quietest_split(audio: np.ndarray, lo: int, hi: int) -> int:
    """Sample index of the lowest-energy 20 ms frame in [lo, hi): most likely a gap between words."""
    frame = SR // 50
    seg = audio[lo:hi]
    n = len(seg) // frame
    if n < 2:
        return hi
    energy = (seg[: n * frame].reshape(n, frame) ** 2).mean(axis=1)
    return lo + int(np.argmin(energy)) * frame + frame // 2


def plan_windows(audio: np.ndarray, regions: list[tuple[int, int]], window_s: float) -> tuple[list[Window], int]:
    """(windows, forced splits). Regions are merged greedily up to window_s; longer regions are force-split."""
    win, ovl, search = int(window_s * SR), int(OVERLAP_S * SR), int(2.0 * SR)
    out: list[Window] = []
    forced = 0
    cur: Window | None = None
    for start, end in regions:
        if end - start > win:  # unbroken speech longer than a window
            if cur:
                out.append(cur)
                cur = None
            s, first = start, True
            while end - s > win:
                cut = _quietest_split(audio, s + win - search, s + win)
                out.append(Window(s, cut, overlap_prev=not first))
                s, first = max(cut - ovl, s + 1), False
                forced += 1
            out.append(Window(s, end, overlap_prev=True))
            continue
        if cur and end - cur.start <= win and start - cur.end <= GAP_SPLIT_S * SR:
            cur.end = end
        else:
            if cur:
                out.append(cur)
            cur = Window(start, end)
    if cur:
        out.append(cur)
    return out, forced


_PUNCT = re.compile(r"[\s\.,!?;:।'\"“”‘’\-–—()]+")


def _norm(word: str) -> str:
    return _PUNCT.sub("", word)


def join_overlap(prev: str, nxt: str, max_words: int = 8) -> str:
    """Drop the start of `nxt` that repeats the end of `prev` (both windows decoded the overlap). Fuzzy, because the
    word cut by the split decodes slightly differently in each window."""
    a, b = prev.split(), nxt.split()
    for k in range(min(max_words, len(a), len(b)), 0, -1):
        x = "".join(_norm(w) for w in a[-k:])
        y = "".join(_norm(w) for w in b[:k])
        if x and y and difflib.SequenceMatcher(None, x, y).ratio() >= 0.75:
            return " ".join(b[k:])
    return nxt


def _suspicious(seg, dur_s: float, token_cap: int) -> str | None:
    n = len(seg.tokens) if seg else 0
    if seg and n >= token_cap - 2:
        return "token cap"
    if dur_s >= MIN_RETRY_AUDIO_S and n / dur_s < MIN_TOK_PER_S:
        return f"{n / dur_s:.1f} tok/s"
    return None


def _decode(orig, pipeline, audio, windows: list[Window], kw: dict) -> tuple[list, object]:
    """One batched decode of `windows`; returns one merged Segment (or None) per window, in order."""
    clips = [{"start": w.start / SR, "end": w.end / SR} for w in windows]
    kw = {k: v for k, v in kw.items() if k not in ("vad_filter", "vad_parameters", "chunk_length", "clip_timestamps")}
    # Overlapping clips can sum to more than the audio, which trips faster-whisper's "VAD removed" bookkeeping
    # (format_timestamp asserts >= 0): pad with silence past the end; no clip reaches into the padding.
    excess = sum(w.end - w.start for w in windows) - len(audio)
    if excess > 0:
        audio = np.concatenate([audio, np.zeros(excess + 1, dtype=np.float32)])
    segs, info = orig(pipeline, audio, clip_timestamps=clips, vad_filter=False, **kw)
    per: list[list] = [[] for _ in windows]
    starts = [c["start"] for c in clips]
    for s in segs:
        i = max((j for j, st in enumerate(starts) if st <= s.start + 0.01), default=0)
        per[i].append(s)
    merged = []
    for group in per:
        if not group:
            merged.append(None)
            continue
        merged.append(dataclasses.replace(
            group[0], end=group[-1].end, text=" ".join(g.text.strip() for g in group).strip(),
            tokens=[t for g in group for t in g.tokens],
            compression_ratio=max(g.compression_ratio for g in group)))
    return merged, info


def transcribe(orig, pipeline, audio, kw: dict):
    """Drop-in for BatchedInferencePipeline.transcribe(audio, vad_filter=True, ...)."""
    from faster_whisper.audio import decode_audio

    if not isinstance(audio, np.ndarray):
        audio = decode_audio(audio, sampling_rate=SR)
    audio = np.asarray(audio, dtype=np.float32)
    token_cap = min(pipeline.model.max_length // 2, pipeline.model.max_length - 3)  # CTranslate2's rule
    regions = _speech_regions(audio)
    windows, forced = plan_windows(audio, regions, WINDOW_S)
    if not windows:
        return orig(pipeline, audio, vad_filter=True, **{k: v for k, v in kw.items() if k != "vad_filter"})
    segs, info = _decode(orig, pipeline, audio, windows, kw)

    retries = []
    for i, (w, seg) in enumerate(zip(windows, segs)):
        why = _suspicious(seg, (w.end - w.start) / SR, token_cap)
        if not why:
            continue
        sub_regions = [(max(s, w.start), min(e, w.end)) for s, e in regions if e > w.start and s < w.end]
        sub, _ = plan_windows(audio, sub_regions or [(w.start, w.end)], RETRY_WINDOW_S)
        sub_segs, _ = _decode(orig, pipeline, audio, sub, kw)
        text = ""
        for sw, ss in zip(sub, sub_segs):
            t = ss.text.strip() if ss else ""
            if text and sw.overlap_prev:
                t = join_overlap(text, t)
            text = f"{text} {t}".strip()
        tokens = [t for ss in sub_segs if ss for t in ss.tokens]
        compression = max((ss.compression_ratio for ss in sub_segs if ss), default=0.0)
        old_n = len(seg.tokens) if seg else 0
        kept = len(tokens) > old_n and compression <= MAX_COMPRESSION
        retries.append({"window": i, "reason": why, "old_tokens": old_n, "new_tokens": len(tokens),
                        "sub_windows": len(sub), "kept": kept})
        if kept:
            base = seg or (next(s for s in sub_segs if s))
            segs[i] = dataclasses.replace(base, start=w.start / SR, end=w.end / SR, text=text, tokens=tokens,
                                          compression_ratio=compression)

    out, text = [], ""
    for w, seg in zip(windows, segs):
        if seg is None or not seg.text.strip():
            continue
        t = join_overlap(text, seg.text.strip()) if w.overlap_prev and text else seg.text.strip()
        text = f"{text} {t}".strip()
        out.append(dataclasses.replace(seg, id=len(out) + 1, text=" " + t))
    _log(audio, windows, forced, segs, retries, token_cap)
    return iter(out), info


def _log(audio, windows, forced, segs, retries, token_cap):
    try:
        from runlog import log
    except ImportError:
        return
    log("whisper_windows", audio_s=round(len(audio) / SR, 3), window_s=WINDOW_S, token_cap=token_cap,
        windows=[{"start_s": round(w.start / SR, 2), "end_s": round(w.end / SR, 2), "overlap_prev": w.overlap_prev,
                  "tokens": len(s.tokens) if s else 0} for w, s in zip(windows, segs)],
        forced_splits=forced, retries=retries)


def install() -> None:
    """Route BatchedInferencePipeline.transcribe(vad_filter=True) through transcribe(). Idempotent; set
    WHISPER_SAFEGUARDS=0 for stock windowing (30 s windows unless WHISPER_WINDOW_S is passed as chunk_length)."""
    from faster_whisper import BatchedInferencePipeline as cls

    if getattr(cls, "_playground_windows", False):
        return
    orig = cls.transcribe
    safeguards = os.environ.get("WHISPER_SAFEGUARDS", "1") != "0"

    def patched(self, audio, *a, **kw):
        if a or not kw.get("vad_filter", True) or kw.get("clip_timestamps"):
            return orig(self, audio, *a, **kw)
        if not safeguards:
            kw.setdefault("chunk_length", int(WINDOW_S))
            return orig(self, audio, **kw)
        return transcribe(orig, self, audio, kw)

    cls.transcribe, cls._playground_windows = patched, True
