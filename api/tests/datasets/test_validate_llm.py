from __future__ import annotations

import pytest

from mlops_api.datasets.validate_llm import validate_llm

_LETTER = {"s": "system", "u": "user", "a": "assistant", "t": "tool"}


def conv(answer: str, question: str = "অর্ডার কোথায়?") -> dict:
    return {"messages": [{"role": "user", "content": question},
                         {"role": "assistant", "content": answer}]}


def codes(issues) -> set[str]:
    return {i.code for i in issues}


def test_llm_valid_sample_has_no_issues(jsonl):
    report, rows = validate_llm(jsonl([conv("চেক করে জানাচ্ছি।")]))
    assert report.ok and not report.warnings and len(rows) == 1
    assert set(rows[0]) == {"sample_id", "messages_json", "tools_json"}
    assert rows[0]["tools_json"] is None and len(rows[0]["sample_id"]) == 16
    assert report.stats["n_samples"] == 1


@pytest.mark.parametrize("roles,ok", [("ua", True), ("sua", True), ("uata", True),
                                      ("au", False), ("uu", False), ("uat", False), ("", False)])
def test_llm_role_order(jsonl, roles, ok):
    msgs = [{"role": _LETTER[c], "content": "ক"} for c in roles]
    report, rows = validate_llm(jsonl([{"messages": msgs}]))
    assert report.ok is ok
    assert (len(rows) == 1) is ok
    if not ok:
        assert codes(report.errors) == {"bad_role_order"}


def test_llm_warnings(jsonl):
    report, _ = validate_llm(jsonl([conv("hello there"), conv("- **দাম** ৳৫০০"),
                                    conv("ধন্যবাদ 😊"), conv("ধন্যবাদ 😊"), conv("ক" * 12001)]))
    assert report.ok
    assert codes(report.warnings) >= {"low_bangla_share", "markdown_in_assistant",
                                      "emoji_in_assistant", "duplicate", "too_long"}
    assert all(w.sample_id for w in report.warnings)


def test_llm_invalid_json_line_is_error(jsonl):
    report, rows = validate_llm(jsonl([conv("ঠিক আছে"), "{not json"]))
    assert codes(report.errors) == {"invalid_json"}
    assert "line 2" in report.errors[0].message
    assert len(rows) == 1


def test_llm_empty_assistant_is_error(jsonl):
    report, _ = validate_llm(jsonl([conv("   ")]))
    assert codes(report.errors) == {"empty_assistant"}


def test_llm_tools_are_kept(jsonl):
    row = conv("ঠিক আছে")
    row["tools"] = [{"type": "function", "function": {"name": "f"}}]
    _, rows = validate_llm(jsonl([row]))
    assert rows[0]["tools_json"] and "function" in rows[0]["tools_json"]
