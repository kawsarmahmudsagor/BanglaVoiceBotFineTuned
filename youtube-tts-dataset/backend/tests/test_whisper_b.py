from dataclasses import dataclass

import numpy as np
import pytest

from ytds.pipeline import whisper_b as wb

SR = 16000


def test_constants():
    assert (wb.MAX_LENGTH, wb.WINDOW_S, wb.RETRY_WINDOW_S, wb.OVERLAP_S) == (880, 10.0, 5.0, 1.0)
    assert (wb.MIN_TOK_PER_S, wb.MIN_RETRY_AUDIO_S, wb.MAX_COMPRESSION) == (12.0, 2.5, 2.4)


def test_join_overlap_drops_repeated_words():
    assert wb.join_overlap("আমি আজ বাজারে গিয়েছিলাম", "গিয়েছিলাম এবং") == "এবং"


def test_join_overlap_mid_word():
    # the split fell inside করতেছিলাম, so the second window starts with its tail
    assert wb.join_overlap("আমি কাজ করতেছিলাম", "আছিলাম তখন") == "তখন"


def test_join_overlap_no_overlap_keeps_text():
    assert wb.join_overlap("আমি আজ", "তুমি কাল") == "তুমি কাল"


def test_suspicious():
    assert wb.suspicious(222, 9.0, 224) == "token cap"
    assert "tok/s" in wb.suspicious(50, 9.0, 224)
    assert wb.suspicious(10, 2.0, 224) is None
    assert wb.suspicious(200, 9.0, 224) is None


def test_plan_windows_force_splits_long_region():
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(23 * SR).astype(np.float32)
    windows, forced = wb.plan_windows(audio, [(0, 23 * SR)], wb.WINDOW_S)
    assert forced == len(windows) - 1 >= 2
    assert all(w.end - w.start <= 10 * SR for w in windows)
    assert [w.overlap_prev for w in windows] == [False] + [True] * (len(windows) - 1)
    assert windows[0].start == 0 and windows[-1].end == 23 * SR


def test_plan_windows_merges_close_regions():
    audio = np.zeros(10 * SR, dtype=np.float32)
    windows, forced = wb.plan_windows(audio, [(0, 2 * SR), (2 * SR + 100, 4 * SR), (8 * SR, 9 * SR)], 10.0)
    assert forced == 0
    assert [(w.start, w.end) for w in windows] == [(0, 4 * SR), (8 * SR, 9 * SR)]


@dataclass
class Seg:
    start: float
    text: str
    tokens: list
    compression_ratio: float


class FakePipeline:
    """One reply per transcribe call: a (text, n_tokens, compression) per clip, in clip order."""

    def __init__(self, replies):
        self.replies, self.calls = replies, []

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        reply = self.replies[len(self.calls) - 1]
        segs = [Seg(c["start"], f" {t} ", [7] * n, cr) for c, (t, n, cr) in zip(kw["clip_timestamps"], reply)]
        return iter(segs), None


PIECES = [{"start_s": 0.0, "end_s": 8.0}, {"start_s": 9.0, "end_s": 12.0}]


@pytest.fixture(autouse=True)
def _no_vad(monkeypatch):
    monkeypatch.setattr(wb, "_speech_regions", lambda audio: [])


def test_decode_pieces_one_batched_call_with_spec_options():
    pipe = FakePipeline([[("ক খ", 200, 1.2), ("গ", 60, 1.1)]])
    out = wb.decode_pieces(pipe, np.zeros(14 * SR, dtype=np.float32), PIECES, token_cap=440)
    assert len(pipe.calls) == 1
    kw = pipe.calls[0]
    assert kw["clip_timestamps"] == [{"start": 0.0, "end": 8.0}, {"start": 9.0, "end": 12.0}]
    assert (kw["vad_filter"], kw["language"], kw["without_timestamps"]) == (False, "bn", True)
    assert (kw["batch_size"], kw["beam_size"]) == (16, 5)
    assert out == [{"text": "ক খ", "tokens": 200, "compression": 1.2, "retry_failed": False},
                   {"text": "গ", "tokens": 60, "compression": 1.1, "retry_failed": False}]


def test_decode_pieces_retry_kept():
    # piece 0: 40 tokens over 8 s = 5 tok/s -> retried as <=5 s sub-windows with a 1 s overlap (silent audio:
    # the quietest frame is the first one searched, so 8 s becomes three sub-windows)
    pipe = FakePipeline([[("ক", 40, 1.2), ("গ", 60, 1.1)], [("চ ছ", 40, 1.8), ("ছ জ", 40, 1.5), ("জ ঝ", 40, 1.0)]])
    out = wb.decode_pieces(pipe, np.zeros(14 * SR, dtype=np.float32), PIECES, token_cap=440)
    assert len(pipe.calls) == 2
    subs = pipe.calls[1]["clip_timestamps"]
    assert len(subs) == 3 and all(c["end"] - c["start"] <= 5.0 for c in subs)
    assert subs[0]["start"] == 0.0 and subs[-1]["end"] == 8.0
    assert out[0] == {"text": "চ ছ জ ঝ", "tokens": 120, "compression": 1.8, "retry_failed": False}
    assert out[1]["text"] == "গ" and out[1]["retry_failed"] is False


def test_decode_pieces_retry_with_repetition_loop_is_dropped():
    pipe = FakePipeline([[("ক", 40, 1.2), ("গ", 60, 1.1)], [("চ ছ", 60, 3.0), ("ছ জ", 60, 1.5), ("জ", 60, 1.0)]])
    out = wb.decode_pieces(pipe, np.zeros(14 * SR, dtype=np.float32), PIECES, token_cap=440)
    assert out[0] == {"text": "ক", "tokens": 40, "compression": 1.2, "retry_failed": True}


def test_decode_pieces_token_cap_triggers_retry():
    pipe = FakePipeline([[("ক", 438, 1.2), ("গ", 60, 1.1)], [("চ", 200, 1.5), ("ছ", 200, 1.5), ("জ", 200, 1.5)]])
    out = wb.decode_pieces(pipe, np.zeros(14 * SR, dtype=np.float32), PIECES, token_cap=440)
    assert out[0]["tokens"] == 600 and out[0]["text"] == "চ ছ জ"


def test_decode_pieces_empty():
    pipe = FakePipeline([])
    assert wb.decode_pieces(pipe, np.zeros(SR, dtype=np.float32), [], token_cap=440) == []
    assert pipe.calls == []


def test_decode_pieces_missing_segment_is_empty_text():
    class Silent(FakePipeline):
        def transcribe(self, audio, **kw):
            self.calls.append(kw)
            return iter([]), None

    pipe = Silent([])
    out = wb.decode_pieces(pipe, np.zeros(3 * SR, dtype=np.float32), [{"start_s": 0.0, "end_s": 2.0}], 440)
    assert out == [{"text": "", "tokens": 0, "compression": 0.0, "retry_failed": False}]
