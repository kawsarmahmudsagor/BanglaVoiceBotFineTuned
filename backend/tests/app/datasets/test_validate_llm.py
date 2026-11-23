from __future__ import annotations

import json

import pytest

from app.datasets.validate_llm import validate_llm

_LETTER = {"s": "system", "u": "user", "a": "assistant", "t": "tool"}


def conv(answer: str, question: str = "অর্ডার কোথায়?") -> dict:
    return {"messages": [{"role": "user", "content": question},
                         {"role": "assistant", "content": answer}]}


def codes(issues) -> set[str]:
    return {i.code for i in issues}


def test_llm_valid_sample_has_no_issues(jsonl):
    report, rows = validate_llm(jsonl([conv("চেক করে জানাচ্ছি।")]))
    assert report.ok and not report.warnings and len(rows) == 1
    assert set(rows[0]) == {"sample_id", "messages_json", "tools_json", "split"}
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


def test_llm_split_field(jsonl):
    report, rows = validate_llm(jsonl([{**conv("ক"), "split": "Validation"}, {**conv("খ"), "split": ""}, conv("গ")]))
    assert report.ok and [r["split"] for r in rows] == ["val", None, None]


def test_unknown_split_value_errors(jsonl):
    report, _ = validate_llm(jsonl([conv("ক"), {**conv("খ"), "split": "hold"}]))
    assert [i.message for i in report.errors] == ["split value 'hold' on line 2: use train, valid or test"]


def _zip(tmp_path, files):
    import zipfile

    z = tmp_path / "u.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for name, lines in files.items():
            zf.writestr(name, "\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n")
    return z


def test_llm_zip_file_layout(tmp_path):
    z = _zip(tmp_path, {"train.jsonl": [conv("ক"), conv("খ")], "dev.jsonl": [conv("গ")], "testing.jsonl": [conv("ঘ")]})
    report, rows = validate_llm(z)
    assert report.ok, report.errors
    assert sorted(r["split"] for r in rows) == ["test", "train", "train", "val"]


def test_llm_zip_without_split_files_errors(tmp_path):
    report, rows = validate_llm(_zip(tmp_path, {"data.jsonl": [conv("ক")]}))
    assert [i.code for i in report.errors] == ["no_split_files"] and rows == []


def test_llm_zip_row_with_split_value_errors(tmp_path):
    report, _ = validate_llm(_zip(tmp_path, {"train.jsonl": [{**conv("ক"), "split": "test"}]}))
    assert [i.code for i in report.errors] == ["split_in_split_file"]
