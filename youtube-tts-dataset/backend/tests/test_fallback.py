import numpy as np
import soundfile as sf

from ytds.llm import LlmUnavailable
from ytds.pipeline import fallback as fb


class FakeGpu:
    def __init__(self, model):
        self.model, self.used = model, []

    def use(self, name):
        self.used.append(name)
        return self.model


class Llm:
    def __init__(self, exc=None):
        self.exc, self.calls = exc, []

    def chat(self, system, user):
        self.calls.append(user)
        if self.exc:
            raise self.exc
        return "আমি ভাত খাই"


def _vocals(tmp_path):
    wav = tmp_path / "v.wav"
    sf.write(str(wav), np.zeros(16000 * 12, dtype="float32"), 16000, subtype="PCM_16")
    return wav


def _clips():
    return [
        {"start_s": 0.0, "end_s": 3.0, "status": "pending_llm", "text_a": "a", "text_b": "b"},
        {"start_s": 4.0, "end_s": 7.0, "status": "pending_llm", "text_a": "a", "text_b": "b", "text_en": "have rice"},
        {"start_s": 8.0, "end_s": 10.0, "status": "needs_review"},
    ]


def test_fallback_sets_translated_needs_review(tmp_path):
    translated = []
    gpu = FakeGpu(lambda audios: translated.append(len(audios)) or ["i eat rice"] * len(audios))
    clips = _clips()
    progress = []
    fb.run_fallback(clips, _vocals(tmp_path), gpu, Llm(), lambda d, t: progress.append((d, t)))
    assert gpu.used == ["whisper_translate"]
    assert translated == [1]  # only the clip lacking text_en
    assert clips[0]["text_en"] == "i eat rice"
    assert clips[1]["text_en"] == "have rice"
    for c in clips[:2]:
        assert c["text_translated"] == c["text_final"] == "আমি ভাত খাই"
        assert c["text_source"] == "translated" and c["status"] == "needs_review"
    assert clips[2]["status"] == "needs_review" and "text_final" not in clips[2]
    assert progress[-1] == (2, 2)


def test_fallback_keeps_pending_llm_when_llm_down(tmp_path):
    gpu = FakeGpu(lambda audios: ["i eat rice"] * len(audios))
    clips = _clips()
    fb.run_fallback(clips, _vocals(tmp_path), gpu, Llm(exc=LlmUnavailable("down")), lambda d, t: None)
    assert clips[0]["status"] == "pending_llm" and clips[0]["text_en"] == "i eat rice"
    assert "text_final" not in clips[0]


def test_retry_skips_whisper_when_all_have_text_en(tmp_path):
    gpu = FakeGpu(lambda audios: (_ for _ in ()).throw(AssertionError("no whisper")))
    clips = [{"start_s": 0.0, "end_s": 3.0, "status": "pending_llm", "text_en": "x"}]
    fb.run_fallback(clips, _vocals(tmp_path), gpu, Llm(), lambda d, t: None)
    assert clips[0]["status"] == "needs_review"
    assert gpu.used == []
