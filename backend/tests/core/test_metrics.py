import pytest

from app.core.metrics import bangla_script_share, cer, has_emoji, has_markdown, sample_wer, wer


def test_wer_known():
    assert wer(["আমি ভাত খাই"], ["আমি খাই"]) == pytest.approx(1 / 3)


def test_wer_corpus_weighted():
    assert wer(["a b", "c d e f"], ["a b", "c x e f"]) == pytest.approx(1 / 6)


def test_cer_ignores_spaces():
    assert cer(["ab cd"], ["ab ce"]) == pytest.approx(1 / 4)


def test_wer_empty_ref_counts_insertions():
    assert sample_wer("", "x y") == 2.0


def test_wer_corpus_empty_ref_uses_max_one_denominator():
    assert wer([""], ["x y"]) == 2.0


def test_cer_corpus_empty_ref_uses_max_one_denominator():
    assert cer([""], ["ab"]) == 2.0


def test_wer_length_mismatch_raises():
    with pytest.raises(ValueError):
        wer(["a"], ["a", "b"])


def test_cer_length_mismatch_raises():
    with pytest.raises(ValueError):
        cer(["a"], [])


def test_script_share():
    assert bangla_script_share("আমি ok") == pytest.approx(3 / 5)


def test_script_share_no_letters_is_zero():
    assert bangla_script_share("১২৩ !!") == 0.0


@pytest.mark.parametrize("t", ["**bold**", "# Title", "- item\n- item", "`code`", "[a](b)"])
def test_markdown(t):
    assert has_markdown(t)


def test_plain_bangla_not_markdown():
    assert not has_markdown("দাম ৫০০ টাকা - ঠিক আছে")


def test_emoji():
    assert has_emoji("ঠিক আছে 😊") and not has_emoji("ঠিক আছে")
