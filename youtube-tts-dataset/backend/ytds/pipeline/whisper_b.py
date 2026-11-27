"""Model B (Whisper turbo-bn fine-tune, CTranslate2) decoding with the Bangla-safe rules of
model-specification/STT/whisper-specs.md (spec section 4.5).

P1: `model.max_length = MAX_LENGTH` lifts the 224-token cap to 440 (never higher).
P2/P3: the pieces are already <= 10 s windows split at pauses / the quietest frame, so each piece is one window of
one batched decode call.
P4: a window that hit the token cap, or is >= MIN_RETRY_AUDIO_S long with fewer than MIN_TOK_PER_S tokens per
second, is decoded again as <= RETRY_WINDOW_S sub-windows (VAD regions inside it, quietest-frame splits, OVERLAP_S
overlap, `join_overlap` dedupe). The retry is kept only if it has more tokens and a compression ratio <=
MAX_COMPRESSION.

The windowing helpers are ported from stt-playground/whisper_chunking.py (origin/main 7e5510e) without its logging
and monkey-patching.
"""
from __future__ import annotations

import dataclasses
import difflib
import re
from typing import Callable

import numpy as np

SR = 16000
MAX_LENGTH = 880
WINDOW_S = 10.0
RETRY_WINDOW_S = 5.0
OVERLAP_S = 1.0
MIN_TOK_PER_S = 12.0
# A pause this long always ends a window. Measured between VAD regions, which are padded by 400 ms each side, so
# 0.2 s here is a ~1 s pause in the audio.
GAP_SPLIT_S = 0.2
MIN_RETRY_AUDIO_S = 2.5  # shorter windows can legitimately hold one or two words
MAX_COMPRESSION = 2.4  # faster-whisper's own repetition threshold
BATCH_SIZE = 16
BEAM_SIZE = 5

DecodeB = Callable[[np.ndarray, list[dict]], list[dict]]


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
    word cut by the split decodes slightly differently in each window.

    Whole-word matches are tried for every length first and the mid-word match only after them: in the
    playground's single pass the mid-word rule could fire at a longer length and drop a word that was not repeated
    ("... বাজারে গিয়েছিলাম" + "গিয়েছিলাম এবং" lost "এবং")."""
    a, b = prev.split(), nxt.split()
    top = min(max_words, len(a), len(b))
    pairs = [(k, "".join(_norm(w) for w in a[-k:]), "".join(_norm(w) for w in b[:k])) for k in range(top, 0, -1)]
    for k, x, y in pairs:
        if x and y and difflib.SequenceMatcher(None, x, y).ratio() >= 0.75:
            return " ".join(b[k:])
    for k, x, y in pairs:
        # The overlap can start mid-word: `nxt` then begins with the tail of a word (করতেছিলাম -> আছিলাম), so
        # compare with the same-length end of `prev` instead.
        if len(y) >= 4 and len(x) > len(y) and difflib.SequenceMatcher(None, x[-len(y):], y).ratio() >= 0.75:
            return " ".join(b[k:])
    return nxt


def suspicious(n_tokens: int, dur_s: float, token_cap: int) -> str | None:
    """Why a window's output looks cut (P4), or None."""
    if n_tokens and n_tokens >= token_cap - 2:
        return "token cap"
    if dur_s >= MIN_RETRY_AUDIO_S and n_tokens / dur_s < MIN_TOK_PER_S:
        return f"{n_tokens / dur_s:.1f} tok/s"
    return None


@dataclasses.dataclass
class _Out:
    text: str = ""
    tokens: int = 0
    compression: float = 0.0


def _decode(pipeline, audio: np.ndarray, windows: list[Window]) -> list[_Out]:
    """One batched decode of `windows` (sorted by start); one merged result per window, in order."""
    clips = [{"start": w.start / SR, "end": w.end / SR} for w in windows]
    # Overlapping clips can sum to more than the audio, which trips faster-whisper's "VAD removed" bookkeeping
    # (format_timestamp asserts >= 0): pad with silence past the end; no clip reaches into the padding.
    excess = sum(w.end - w.start for w in windows) - len(audio)
    if excess > 0:
        audio = np.concatenate([audio, np.zeros(excess + 1, dtype=np.float32)])
    segs, _ = pipeline.transcribe(audio, clip_timestamps=clips, vad_filter=False, language="bn",
                                  without_timestamps=True, batch_size=BATCH_SIZE, beam_size=BEAM_SIZE)
    per: list[list] = [[] for _ in windows]
    starts = [c["start"] for c in clips]
    for s in segs:
        i = max((j for j, st in enumerate(starts) if st <= s.start + 0.01), default=0)
        per[i].append(s)
    return [_Out(" ".join(g.text.strip() for g in group).strip(), sum(len(g.tokens) for g in group),
                 max(g.compression_ratio for g in group)) if group else _Out() for group in per]


def _retry(pipeline, audio: np.ndarray, w: Window) -> _Out:
    """P4: the window again as <= RETRY_WINDOW_S sub-windows built from its VAD regions."""
    regions = [(w.start + s, w.start + e) for s, e in _speech_regions(audio[w.start:w.end])]
    regions = [(max(s, w.start), min(e, w.end)) for s, e in regions if e > s]
    sub, _ = plan_windows(audio, regions or [(w.start, w.end)], RETRY_WINDOW_S)
    outs = _decode(pipeline, audio, sub)
    text = ""
    for sw, o in zip(sub, outs):
        t = o.text
        if text and sw.overlap_prev:
            t = join_overlap(text, t)
        text = f"{text} {t}".strip()
    return _Out(text, sum(o.tokens for o in outs), max((o.compression for o in outs if o.tokens), default=0.0))


def decode_pieces(pipeline, audio16k: np.ndarray, pieces: list[dict], token_cap: int) -> list[dict]:
    """Each piece (start_s/end_s on `audio16k`) as one window of one batched decode, then P4 retries.
    Per piece: {text, tokens, compression, retry_failed}."""
    if not pieces:
        return []
    audio = np.asarray(audio16k, dtype=np.float32)
    order = sorted(range(len(pieces)), key=lambda i: pieces[i]["start_s"])
    windows = [Window(max(0, int(pieces[i]["start_s"] * SR)), min(len(audio), int(pieces[i]["end_s"] * SR)))
               for i in order]
    outs = _decode(pipeline, audio, windows)
    failed = [False] * len(windows)
    for k, (w, o) in enumerate(zip(windows, outs)):
        if not suspicious(o.tokens, (w.end - w.start) / SR, token_cap):
            continue
        r = _retry(pipeline, audio, w)
        if r.tokens > o.tokens and r.compression <= MAX_COMPRESSION:
            outs[k] = r
        else:
            failed[k] = True
    result: list[dict] = [{}] * len(pieces)
    for k, i in enumerate(order):
        o = outs[k]
        result[i] = {"text": o.text, "tokens": o.tokens, "compression": o.compression, "retry_failed": failed[k]}
    return result


def load_whisper(path: str) -> DecodeB:
    """GPU loader body for model B: a CT2 folder or repo id, float16 on CUDA, P1 applied."""
    import torch  # noqa: F401  on Windows this loads torch/lib's cuBLAS/cuDNN DLLs, which CTranslate2 needs
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    model = WhisperModel(path, device="cuda", compute_type="float16")
    model.max_length = MAX_LENGTH
    token_cap = min(MAX_LENGTH // 2, MAX_LENGTH - 3)  # CTranslate2's rule: 440
    pipeline = BatchedInferencePipeline(model=model)
    return lambda audio16k, pieces: decode_pieces(pipeline, audio16k, pieces, token_cap)
