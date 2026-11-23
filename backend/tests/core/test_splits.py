import pytest
from pydantic import ValidationError

from app.core.splits import EmptySplitError, SplitConfig, SplitRatios, assign, normalize_split


def rows(labels):
    return [{"sample_id": f"{i:016x}", "split": lab} for i, lab in enumerate(labels)]


def test_no_labels_90_5_5():
    out = assign(rows([None] * 20000), SplitConfig())
    assert abs(out.count("train") / 20000 - 0.9) < 0.01 and abs(out.count("val") / 20000 - 0.05) < 0.01
    assert abs(out.count("test") / 20000 - 0.05) < 0.01


def test_all_given_kept():
    labels = ["train", "val", "test"] * 10
    assert assign(rows(labels), SplitConfig()) == labels


def test_train_and_test_given_carves_val_from_train():
    labels = ["train"] * 9000 + ["test"] * 1000
    out = assign(rows(labels), SplitConfig())
    assert out[9000:] == ["test"] * 1000
    assert abs(out[:9000].count("val") / 9000 - 0.05 / 0.95) < 0.01 and "test" not in out[:9000]


def test_only_test_given_rest_split_90_5():
    labels = ["test"] * 1000 + [None] * 9000
    out = assign(rows(labels), SplitConfig())
    assert out[:1000] == ["test"] * 1000
    assert abs(out[1000:].count("train") / 9000 - 0.9 / 0.95) < 0.01


def test_ignore_dataset_splits():
    out = assign(rows(["test"] * 10000), SplitConfig(use_dataset_splits=False))
    assert abs(out.count("train") / 10000 - 0.9) < 0.01


def test_deterministic_and_order_independent():
    r = rows([None] * 500)
    a = dict(zip([x["sample_id"] for x in r], assign(r, SplitConfig())))
    rr = list(reversed(r))
    assert dict(zip([x["sample_id"] for x in rr], assign(rr, SplitConfig()))) == a


def test_empty_split_error():
    with pytest.raises(EmptySplitError, match="split 'val' is empty: add data or change datasets.split.ratios"):
        assign(rows([None]), SplitConfig())


def test_zero_ratio_split_may_be_empty():
    assign(rows([None] * 10), SplitConfig(ratios=SplitRatios(train=0.9, val=0.1, test=0.0)))


@pytest.mark.parametrize("bad", [dict(train=0.5, val=0.2, test=0.2), dict(train=0, val=0.5, test=0.5),
                                 dict(train=1.1, val=-0.1, test=0),
                                 dict(train=0.9, val=float("nan"), test=0.05),
                                 dict(train=float("nan"), val=0.05, test=0.05),
                                 dict(train=float("inf"), val=0.0, test=0.0),
                                 dict(train=0.9, val=float("-inf"), test=0.1)])
def test_ratio_validation(bad):
    with pytest.raises(ValidationError):
        SplitRatios(**bad)


@pytest.mark.parametrize("v,exp", [("Validation", "val"), ("dev", "val"), ("TRAINING", "train"),
                                   ("testing", "test"), ("", None), (None, None)])
def test_normalize_split(v, exp):
    assert normalize_split(v) == exp


def test_normalize_split_unknown():
    with pytest.raises(ValueError):
        normalize_split("hold")
