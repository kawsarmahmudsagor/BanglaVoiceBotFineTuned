from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .report import Issue, Report

_ROLE_LETTER = {"system": "s", "user": "u", "assistant": "a", "tool": "t"}
_ROLE_RE = re.compile(r"^s?(u(a(t+a)*))+$")
_MARKDOWN_LINE_RE = re.compile(r"^\s*(#{1,6}\s|[-*]\s|\d+\.\s|\|)")
_MAX_CHARS = 12_000


def _is_emoji(ch: str) -> bool:
    o = ord(ch)
    return 0x1F300 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF


def _has_markdown(text: str) -> bool:
    if "**" in text or "`" in text:
        return True
    return any(_MARKDOWN_LINE_RE.match(line) for line in text.split("\n"))


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def validate_llm(path: Path) -> tuple[Report, list[dict]]:
    report = Report()
    rows: list[dict] = []
    seen: set[str] = set()
    try:
        # utf-8-sig tolerates an Excel/Notepad BOM; file iteration splits only on line breaks
        with open(path, encoding="utf-8-sig") as fh:
            for lineno, line in enumerate(fh, start=1):
                if line.strip():
                    _check_line(report, rows, seen, lineno, line)
    except UnicodeDecodeError:
        report.errors.append(Issue("invalid_json", "file is not valid UTF-8"))
        rows = []
    report.stats["n_samples"] = len(rows)
    return report, rows


def _check_line(report: Report, rows: list[dict], seen: set[str], lineno: int, line: str) -> None:
    def err(code: str, msg: str) -> None:
        report.errors.append(Issue(code, f"line {lineno}: {msg}"))

    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        err("invalid_json", str(e))
        return
    messages = obj.get("messages") if isinstance(obj, dict) else None
    if not isinstance(messages, list) or not all(
        isinstance(m, dict) and isinstance(m.get("role"), str) for m in messages
    ):
        err("invalid_json", "expected an object with a 'messages' list of {role, content}")
        return
    if any(m.get("content") is not None and not isinstance(m["content"], str) for m in messages):
        err("invalid_json", "message content must be a string")
        return

    letters = "".join(_ROLE_LETTER.get(m["role"], "?") for m in messages)
    if not _ROLE_RE.match(letters):
        err("bad_role_order", f"role sequence {[m['role'] for m in messages]} is not valid")
        return
    assistants = [m for m in messages if m["role"] == "assistant"]
    if any(not (m.get("content") or "").strip() and not m.get("tool_calls") for m in assistants):
        err("empty_assistant", "assistant message has no content")
        return

    messages_json = _canon(messages)
    tools = obj.get("tools")
    tools_json = _canon(tools) if tools else None
    sample_id = hashlib.sha256(_canon({"m": messages, "t": tools}).encode("utf-8")).hexdigest()[:16]
    rows.append({"sample_id": sample_id, "messages_json": messages_json, "tools_json": tools_json})

    def warn(code: str, msg: str) -> None:
        report.warnings.append(Issue(code, f"line {lineno}: {msg}", sample_id))

    text = "\n".join(m.get("content") or "" for m in assistants)
    alpha = [c for c in text if c.isalpha()]
    if alpha:
        share = sum(1 for c in alpha if "ঀ" <= c <= "৿") / len(alpha)
        if share < 0.5:
            warn("low_bangla_share", f"Bangla share {share:.0%}")
    if _has_markdown(text):
        warn("markdown_in_assistant", "assistant text contains markdown")
    if any(_is_emoji(c) for c in text):
        warn("emoji_in_assistant", "assistant text contains emoji")
    if messages_json in seen:
        warn("duplicate", "identical to an earlier sample")
    seen.add(messages_json)
    if sum(len(m.get("content") or "") for m in messages) > _MAX_CHARS:
        warn("too_long", f"more than {_MAX_CHARS} characters")
