"""Chat-dataset converter: tokenized rows with assistant-only labels (JSONL)."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from app.core.backend_kit import max_skip_share
from app.core.contract import write_prepared_refs
from app.core.prepare import BuildResult, PrepSpec, Skip, ensure_prepared

from ..models import MODELS

IGNORE = -100


def chat_spec(cfg) -> PrepSpec:
    info = MODELS.get(cfg.base_model)
    return PrepSpec(family="chat", version=1, params={
        "base_model": cfg.base_model,
        "chat_template": info.chat_template if info else None,
        "max_seq_len": int(cfg.hparams.get("max_seq_len", 2048)),
    })


def _ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


NOT_STABLE = "chat template is not prefix-stable"
NO_ASSISTANT = "no assistant turn"


def _candidates(content: str) -> list[str]:
    """Forms an assistant content may take once a template rewrites it (e.g. Qwen3 drops <think> blocks
    from earlier turns and strips whitespace)."""
    out = [content, content.strip()]
    if "</think>" in content:
        out.append(content.rsplit("</think>", 1)[1].strip())
    return [c for i, c in enumerate(out) if c and c not in out[:i]]


def _encode(tokenizer, messages: list[dict], tools) -> tuple[list[int], list[int]] | str:
    """(input_ids, labels), or a skip reason.

    Fast path: when every prefix render is a literal prefix of the full render, each assistant span is
    the token-length difference between the generation-prompt prefix and the prefix including the turn.
    Otherwise (templates that rewrite earlier turns) the full text is tokenized once with offsets and the
    assistant contents are located in it. In the fast path the "
" after an end marker is labelled when
    it is a separate token (harmless; not masked)."""
    def render(msgs, gen=False):
        kw = {"add_generation_prompt": True} if gen else {}
        return tokenizer.apply_chat_template(msgs, tools=tools, tokenize=False, **kw)

    turns = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if not turns:
        return NO_ASSISTANT
    full = render(messages)
    pairs = []
    stable = True
    for i in turns:
        pre, upto = render(messages[:i], gen=True), render(messages[: i + 1])
        if not (full.startswith(upto) and upto.startswith(pre)):
            stable = False
            break
        pairs.append((pre, upto))
    if stable:
        ids = _ids(tokenizer, full)
        labels = [IGNORE] * len(ids)
        for pre, upto in pairs:
            start, end = len(_ids(tokenizer, pre)), len(_ids(tokenizer, upto))
            assert start < end <= len(ids), (start, end, len(ids))
            labels[start:end] = ids[start:end]
        return ids, labels
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = list(enc["input_ids"]), enc["offset_mapping"]
    eos = getattr(tokenizer, "eos_token", None)
    spans = []
    cursor = 0
    for i in turns:
        for cand in _candidates(messages[i].get("content") or ""):
            at = full.find(cand, cursor)
            if at >= 0:
                break
        else:
            return NOT_STABLE
        end = at + len(cand)
        pos = end
        while pos < len(full) and full[pos].isspace():
            pos += 1
        if eos and full.startswith(eos, pos):
            end = pos + len(eos)
        spans.append((at, end))
        cursor = end
    labels = [IGNORE] * len(ids)
    for j, (s, e) in enumerate(offs):
        if any(s >= a and e <= b and e > s for a, b in spans):
            labels[j] = ids[j]
    return ids, labels


def build_chat(tokenizer, max_seq_len: int) -> Callable[[dict[str, list[dict]], Path], BuildResult]:
    def build(rows: dict[str, list[dict]], out: Path) -> BuildResult:
        counts: dict[str, int] = {}
        skipped: list[Skip] = []
        files: list[str] = []
        for split, items in rows.items():
            kept = 0
            with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
                for row in items:
                    enc = _encode(tokenizer, row["messages"], row.get("tools"))
                    if isinstance(enc, str):
                        skipped.append(Skip(sample_id=str(row.get("sample_id", "")), split=split, reason=enc))
                        continue
                    ids, labels = enc
                    if len(ids) > max_seq_len:
                        skipped.append(Skip(sample_id=str(row.get("sample_id", "")), split=split,
                                            reason=f"{len(ids)} tokens > max_seq_len {max_seq_len}"))
                        continue
                    f.write(json.dumps({"input_ids": ids, "labels": labels}) + "\n")
                    kept += 1
            counts[split] = kept
            files.append(f"{split}.jsonl")
        return BuildResult(counts=counts, skipped=skipped, files=files)

    return build


def _load_tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def prepare_llm(rd, cfg, rep) -> int:
    info = MODELS.get(cfg.base_model)
    if info is not None and info.gated:
        from app.core.backend_kit.hf_access import check_hf_access

        check_hf_access(cfg.base_model)
    tok = _load_tokenizer(cfg.base_model)
    spec = chat_spec(cfg)
    build = build_chat(tok, spec.params["max_seq_len"])
    share = max_skip_share(cfg)
    refs = []
    for role, key, splits in (("train", "train", ["train", "val"]), ("eval", "eval", None)):
        ds = cfg.datasets.get(key)
        if ds is None:
            continue
        pi = ensure_prepared(cfg, ds, spec, build, splits=splits, max_skip_share=share)
        refs.append({"role": role, **pi.model_dump()})
    write_prepared_refs(rd, refs)
    return 0
