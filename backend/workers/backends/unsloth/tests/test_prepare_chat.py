import json

import pytest

from app.core.prepare import PrepSpec
from mlops_backend.prepare.chat import build_chat, chat_spec, prepare_llm


class FakeTok:
    def __init__(self):
        self.tools_seen = []

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False, **kw):
        self.tools_seen.append(tools)
        out = "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
        if add_generation_prompt:
            out += "<|assistant|>"
        return out

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **kw):
        out = {"input_ids": [ord(c) for c in text]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class RewritingTok(FakeTok):
    """Template wraps non-final assistant turns in parens, so prefixes are not stable."""

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False, **kw):
        last = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
        out = "".join(
            f"<|{m['role']}|>" + (f"({m['content']})" if m["role"] == "assistant" and i != last else m["content"])
            for i, m in enumerate(messages))
        return out + ("<|assistant|>" if add_generation_prompt else "")


class DroppingTok(FakeTok):
    """Template drops earlier assistant content entirely."""

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False, **kw):
        last = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
        out = "".join(f"<|{m['role']}|>" + ("" if m["role"] == "assistant" and i != last else m["content"])
                      for i, m in enumerate(messages))
        return out + ("<|assistant|>" if add_generation_prompt else "")


def _build(rows, max_len=1000, tok=None, tmp=None):
    tok = tok or FakeTok()
    return tok, build_chat(tok, max_len)({"train": rows}, tmp)


def _read(tmp, split="train"):
    return [json.loads(line) for line in (tmp / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]


def _row(msgs, sid="s0", tools=None):
    return {"sample_id": sid, "split": "train", "messages": msgs, "tools": tools}


def _labeled(tok, rec):
    return tok.decode([t for t in rec["labels"] if t != -100])


def test_labels_mask_non_assistant(tmp_path):
    tok, res = _build([_row([{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}])],
                      tmp=tmp_path)
    rec = _read(tmp_path)[0]
    assert len(rec["input_ids"]) == len(rec["labels"])
    assert _labeled(tok, rec) == "world"
    assert res.counts == {"train": 1}


def test_multi_turn_masks_each_assistant_turn(tmp_path):
    msgs = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]
    tok, _ = _build([_row(msgs)], tmp=tmp_path)
    assert _labeled(tok, _read(tmp_path)[0]) == "a1a2"


def test_overlong_rows_skipped_with_reason(tmp_path):
    rows = [_row([{"role": "user", "content": "x" * 50}, {"role": "assistant", "content": "y"}], sid="long"),
            _row([{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}], sid="ok")]
    _, res = _build(rows, max_len=40, tmp=tmp_path)
    assert res.counts == {"train": 1}
    assert len(res.skipped) == 1
    sk = res.skipped[0]
    assert sk.sample_id == "long" and sk.split == "train"
    n = len("<|user|>" + "x" * 50 + "<|assistant|>y")
    assert sk.reason == f"{n} tokens > max_seq_len 40"


def test_tools_passed_to_template(tmp_path):
    tools = [{"type": "function", "function": {"name": "f"}}]
    tok, _ = _build([_row([{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}],
                          tools=tools)], tmp=tmp_path)
    assert tok.tools_seen and all(t == tools for t in tok.tools_seen)


_TWO = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]


def test_unstable_template_falls_back_to_offsets(tmp_path):
    tok, res = _build([_row(_TWO)], tok=RewritingTok(), tmp=tmp_path)
    assert res.counts == {"train": 1} and not res.skipped
    assert _labeled(tok, _read(tmp_path)[0]) == "a1a2"


def test_unstable_template_content_not_found_skipped(tmp_path):
    _, res = _build([_row(_TWO)], tok=DroppingTok(), tmp=tmp_path)
    assert res.counts == {"train": 0}
    assert [s.reason for s in res.skipped] == ["chat template is not prefix-stable"]


def test_no_assistant_turn_skipped(tmp_path):
    _, res = _build([_row([{"role": "user", "content": "q"}])], tmp=tmp_path)
    assert [s.reason for s in res.skipped] == ["no assistant turn"]


def test_gated_access_error_surfaces(run_dir, monkeypatch):
    from app.core.contract import read_config
    from app.core.prepare import PrepareError

    rd = run_dir("llm-sft", "google/gemma-3-4b-it")

    def deny(repo):
        raise PrepareError(f"no access to {repo}: accept its terms on Hugging Face and set HF_TOKEN")

    monkeypatch.setattr("app.core.backend_kit.hf_access.check_hf_access", deny)
    with pytest.raises(PrepareError) as ei:
        prepare_llm(rd, read_config(rd), None)
    assert str(ei.value) == ("no access to google/gemma-3-4b-it: accept its terms on Hugging Face "
                             "and set HF_TOKEN")


def test_chat_spec():
    cfg = type("C", (), {"base_model": "Qwen/Qwen3-4B-Instruct-2507", "hparams": {"max_seq_len": 512}})()
    spec = chat_spec(cfg)
    assert isinstance(spec, PrepSpec)
    assert spec.family == "chat" and spec.version == 1
    assert spec.params == {"base_model": "Qwen/Qwen3-4B-Instruct-2507", "chat_template": "qwen3",
                           "max_seq_len": 512}


def test_prepare_llm_writes_refs(run_dir, monkeypatch):
    from app.core.contract import read_prepared_refs

    rd = run_dir("llm-sft", "Qwen/Qwen3-4B-Instruct-2507", hparams={"max_seq_len": 256})
    monkeypatch.setattr("mlops_backend.prepare.chat._load_tokenizer", lambda name: FakeTok())
    from app.core.contract import read_config

    assert prepare_llm(rd, read_config(rd), None) == 0
    refs = read_prepared_refs(rd)
    assert [r["role"] for r in refs] == ["train", "eval"]
    assert refs[0]["spec"]["family"] == "chat"
    assert refs[0]["counts"] == {"train": 2, "val": 1}


@pytest.mark.heavy
def test_qwen3_real_template_masks_assistant_only(tmp_path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    msgs = [{"role": "system", "content": "be terse SYSTEMTEXT"},
            {"role": "user", "content": "hi there"},
            {"role": "assistant", "content": "<think>x</think>hello friend"},
            {"role": "user", "content": "and again"},
            {"role": "assistant", "content": "second reply"}]
    build_chat(tok, 512)({"train": [_row(msgs)]}, tmp_path)
    rec = _read(tmp_path)[0]
    text = tok.decode([t for t in rec["labels"] if t != -100])
    assert text.index("hello friend") < text.index("second reply")
    assert text.count("<|im_end|>") == 2
    for bad in ("<|im_start|>user", "SYSTEMTEXT", "hi there", "and again"):
        assert bad not in text
