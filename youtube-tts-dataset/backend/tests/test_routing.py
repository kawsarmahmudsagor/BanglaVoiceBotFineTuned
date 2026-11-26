from ytds.config import Thresholds
from ytds.pipeline.routing import cer, route

SENTENCE = "আমার সোনার বাংলা আমি তোমায় ভালোবাসি"
REF20 = "কখগঘঙচছজঝঞটঠডঢণতথদধন"  # 20 code points, no combining marks


def _swap(n: int) -> str:
    """REF20 with its first n characters replaced by one that never occurs in it (n substitutions)."""
    return "য" * n + REF20[n:]


def test_cer_identical_is_zero():
    assert cer(SENTENCE, SENTENCE) == 0.0


def test_cer_both_empty_is_zero():
    assert cer("", "") == 0.0
    assert cer(" । ", "") == 0.0


def test_cer_ignores_spaces_and_punctuation():
    assert cer(SENTENCE, SENTENCE.replace(" ", "") + "।") == 0.0


def test_cer_empty_ref_divides_by_one():
    assert cer("", "কখ") == 2.0


def test_route_boundaries():
    th = Thresholds()
    assert len(REF20) == 20
    assert route(REF20, _swap(1), th) == ("auto", 0.05)  # 1 / 20, exactly cer_auto
    assert route(REF20, _swap(2), th) == ("flagged", 0.1)
    assert route(REF20, _swap(6), th) == ("flagged", 0.3)  # 6 / 20, exactly cer_fallback
    assert route(REF20, _swap(7), th) == ("fallback", 0.35)


def test_route_empty_a_is_fallback():
    assert route("", "কিছু একটা", Thresholds()) == ("fallback", 1.0)
    assert route(" । ", "", Thresholds()) == ("fallback", 1.0)


def test_route_latin_and_digits_no_crash():
    status, c = route("iPhone 15 এর দাম", "আইফোন ১৫ এর দাম", Thresholds())
    assert status in ("auto", "flagged", "fallback")
    assert c > 0.0
