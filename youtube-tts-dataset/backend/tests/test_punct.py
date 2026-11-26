from ytds.llm import LlmUnavailable
from ytds.pipeline.punct import punctuate, words_unchanged

TEXT = "আমি ভাত খাই তুমি কি খাও"


class Llm:
    def __init__(self, out=None, exc=None):
        self.out, self.exc, self.calls = out, exc, 0

    def chat(self, system, user):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.out


def test_guard_accepts_punct_only():
    assert words_unchanged(TEXT, "আমি ভাত খাই। তুমি কি খাও?")
    assert words_unchanged(TEXT, "“আমি ভাত খাই”, তুমি কি খাও?")


def test_guard_rejects_changed_word():
    assert not words_unchanged(TEXT, "আমি ভাত খাব। তুমি কি খাও?")


def test_guard_rejects_dropped_word():
    assert not words_unchanged(TEXT, "আমি ভাত খাই। তুমি কি?")


def test_punctuate_accepts_and_flags():
    out, no_punct = punctuate(TEXT, Llm("আমি ভাত খাই। তুমি কি খাও?"))
    assert out == "আমি ভাত খাই। তুমি কি খাও?" and no_punct is False


def test_punctuate_rejected_output_returns_original():
    assert punctuate(TEXT, Llm("আমি ভাত খাব")) == (TEXT, True)


def test_punctuate_falls_back_on_unavailable():
    assert punctuate(TEXT, Llm(exc=LlmUnavailable("down"))) == (TEXT, True)


def test_punctuate_empty_skips_llm():
    llm = Llm("x")
    assert punctuate("", llm) == ("", True)
    assert llm.calls == 0
