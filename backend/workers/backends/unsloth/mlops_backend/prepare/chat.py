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


def _encode(tokenizer, messages: list[dict], tools) -> tuple[list[int], list[int]]:
    """(input_ids, labels). Each assistant span runs from the end of the generation-prompt prefix to the
    end of the prefix that includes the turn. Prefixes are tokenized separately, so a real tokenizer that
    merges tokens across a turn boundary can shift a span by a token; chat templates put special tokens
    at these boundaries, which prevents merging in practice."""
    def render(msgs, gen=False):
        kw = {"add_generation_prompt": True} if gen else {}
        return tokenizer.apply_chat_template(msgs, tools=tools, tokenize=False, **kw)

    ids = _ids(tokenizer, render(messages))
    labels = [IGNORE] * len(ids)
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        start = len(_ids(tokenizer, render(messages[:i], gen=True)))
        end = len(_ids(tokenizer, render(messages[: i + 1])))
        for j in range(min(start, len(ids)), min(end, len(ids))):
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
                    ids, labels = _encode(tokenizer, row["messages"], row.get("tools"))
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
