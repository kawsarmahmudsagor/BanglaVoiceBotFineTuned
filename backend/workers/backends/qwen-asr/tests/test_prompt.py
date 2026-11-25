"""Prompt format and label masking (spec 17.4).

Light tests use FakeProcessor, which mirrors what the real Qwen3ASRProcessor does with this model's chat template
(checked on the VM with transformers 5.24.1, see test_real_* below):
- the template is static apart from one `<|audio_start|><|audio_pad|><|audio_end|>` block in the user turn;
  with continue_final_message=True the rendering stops right after the assistant text (no `<|im_end|>`);
- `<asr_text>`, `<|im_end|>` and the audio tokens are single special tokens; other text tokenizes normally;
- calling the processor expands `<|audio_pad|>` to one token per audio frame group (a count that depends only
  on the clip length) and pads input_ids/attention_mask on `padding_side`.
The tests marked real_processor load the real processor from the local HF cache (no weights) and are skipped
when it is not there.
"""
import re

import numpy as np
import pytest

from mlops_backend.prompt import (
    IGNORE, MARKER, PREFIX, QwenAsrCollator, conversation, generation_inputs, mask_labels, max_new_tokens,
    prompt_text, split_transcript, training_text,
)

MODEL = "SayedShaun/Qwen3-ASR-0.6B-Bengali"
SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|audio_start|>", "<|audio_pad|>", "<|audio_end|>",
            MARKER]


class FakeTokenizer:
    eos_token = "<|im_end|>"
    pad_token = "<|endoftext|>"

    def __init__(self):
        self.special = {t: 1000 + i for i, t in enumerate(SPECIALS)}
        self.inv = {v: k for k, v in self.special.items()}
        self.pad_token_id = self.special[self.pad_token]
        self._re = re.compile("|".join(re.escape(t) for t in SPECIALS))

    def convert_tokens_to_ids(self, tok):
        return self.special[tok]

    def encode(self, text):
        ids, pos = [], 0
        for m in self._re.finditer(text):
            ids += [ord(c) for c in text[pos:m.start()]]
            ids.append(self.special[m.group()])
            pos = m.end()
        return ids + [ord(c) for c in text[pos:]]

    def decode(self, ids):
        return "".join(self.inv.get(i, chr(i)) if i >= 0 else "" for i in ids)


class FakeProcessor:
    """Real template shape; audio pad expanded to len(audio) // 4000 tokens (at least 1)."""

    def __init__(self, template_ok=True):
        self.tokenizer = FakeTokenizer()
        self.template_ok = template_ok

    def apply_chat_template(self, conv, tokenize=False, continue_final_message=False):
        assert tokenize is False
        s = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
        for m in conv:
            if m["role"] == "assistant":
                text = "".join(c["text"] for c in m["content"]) if self.template_ok else ""
                s += "<|im_start|>assistant\n" + text + ("" if continue_final_message else "<|im_end|>\n")
        return s

    def __call__(self, text, audio, sampling_rate, padding, padding_side, return_tensors):
        assert sampling_rate == 16000 and padding is True and return_tensors == "pt"
        rows = []
        for t, a in zip(text, audio):
            t = t.replace("<|audio_pad|>", "<|audio_pad|>" * max(1, len(a) // 4000))
            rows.append(self.tokenizer.encode(t))
        width = max(len(r) for r in rows)
        pad = self.tokenizer.pad_token_id
        ids, mask = [], []
        for r in rows:
            fill = width - len(r)
            if padding_side == "right":
                ids.append(r + [pad] * fill)
                mask.append([1] * len(r) + [0] * fill)
            else:
                ids.append([pad] * fill + r)
                mask.append([0] * fill + [1] * len(r))
        return {"input_ids": ids, "attention_mask": mask, "input_features": [[0.0]] * len(rows),
                "input_features_mask": [[1]] * len(rows)}


def loader_of(seconds):
    return lambda path: np.zeros(int(16000 * seconds[path]), dtype="float32")


def test_prefix_and_conversation():
    assert PREFIX == "language Bengali<asr_text>"
    conv = conversation("x.wav")
    assert conv[0] == {"role": "user", "content": [{"type": "audio", "path": "x.wav"}]}
    assert conv[1]["role"] == "assistant" and conv[1]["content"][0]["text"] == PREFIX


def test_prompt_text_ends_with_prefix():
    p = prompt_text(FakeProcessor())
    assert p.endswith("<|im_start|>assistant\n" + PREFIX)
    with pytest.raises(ValueError, match="does not end with the prefix"):
        prompt_text(FakeProcessor(template_ok=False))


def test_mask_labels_masks_through_marker_and_padding():
    m = 7
    ids = [[1, 2, m, 4, 5, 9], [1, m, 4, 9, 0, 0]]
    att = [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]
    assert mask_labels(ids, att, m) == [[IGNORE, IGNORE, IGNORE, 4, 5, 9], [IGNORE, IGNORE, 4, 9, IGNORE, IGNORE]]


def test_mask_labels_first_marker_only_and_missing_marker():
    assert mask_labels([[7, 3, 7, 5]], [[1, 1, 1, 1]], 7) == [[IGNORE, 3, 7, 5]]
    with pytest.raises(ValueError, match="not found"):
        mask_labels([[1, 2, 3]], [[1, 1, 1]], 7)


def test_collator_loss_only_on_transcript_and_eos():
    proc = FakeProcessor()
    tok = proc.tokenizer
    col = QwenAsrCollator(proc, loader=loader_of({"a": 1.0, "b": 2.5}))
    feats = [{"audio_path": "a", "text": "আমি যাব"}, {"audio_path": "b", "text": "তুমি এসো কাল"}]
    b = col(feats)
    assert set(b) == {"input_ids", "attention_mask", "input_features", "input_features_mask", "labels"}
    for row, labels, att, f in zip(b["input_ids"], b["labels"], b["attention_mask"], feats):
        kept = [t for t in labels if t != IGNORE]
        assert tok.decode(kept) == f["text"] + "<|im_end|>"  # transcript + EOS, nothing else
        cut = row.index(tok.convert_tokens_to_ids(MARKER))
        assert all(t == IGNORE for t in labels[:cut + 1])  # system, user + audio, assistant header, prefix
        assert tok.decode(row[:cut + 1]).endswith(PREFIX)
        assert all(lab == IGNORE for lab, a in zip(labels, att) if not a)  # padding
        assert att[0] == 1  # right padding: real tokens first
    # different clip lengths -> different audio token counts, and the boundary moves with them
    n_audio = [r.count(tok.convert_tokens_to_ids("<|audio_pad|>")) for r in b["input_ids"]]
    assert n_audio[0] != n_audio[1]


def test_training_text_and_generation_inputs_left_pad():
    proc = FakeProcessor()
    p = prompt_text(proc)
    assert training_text(p, "আমি", "<|im_end|>") == p + "আমি<|im_end|>"
    b = generation_inputs(proc, p, [np.zeros(16000, "float32"), np.zeros(40000, "float32")])
    assert b["attention_mask"][0][0] == 0 and b["attention_mask"][1][0] == 1  # left padded
    assert all(proc.tokenizer.decode(r).endswith(PREFIX) for r in b["input_ids"])


def test_split_transcript_and_max_new_tokens():
    assert split_transcript("language Bengali<asr_text> আমি যাব ") == "আমি যাব"
    assert split_transcript(" আমি যাব") == "আমি যাব"
    assert split_transcript("x<asr_text>a<asr_text>b") == "a<asr_text>b"
    assert max_new_tokens(2.0) == 76
    assert max_new_tokens(30) == 916


# ---------------------------------------------------------------- real processor (local HF cache, no weights)

@pytest.fixture(scope="module")
def real_processor():
    pytest.importorskip("transformers")
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{MODEL} processor not in the local HF cache ({type(e).__name__})")


def test_real_training_text_matches_chat_template(real_processor):
    p = prompt_text(real_processor)
    text = "আমি ভাত খাই"
    full = [conversation()[0], {"role": "assistant", "content": [{"type": "text", "text": PREFIX + text}]}]
    rendered = real_processor.apply_chat_template(full, tokenize=False)
    # the full chat-template rendering is our training text plus the template's trailing newline
    assert rendered == training_text(p, text, real_processor.tokenizer.eos_token) + "\n"
    from transformers import GenerationConfig

    # the EOS we train on is one that generation stops at
    eos = GenerationConfig.from_pretrained(MODEL, local_files_only=True).eos_token_id
    assert real_processor.tokenizer.eos_token_id in (eos if isinstance(eos, list) else [eos])


def test_real_collator_labels(real_processor):
    tok = real_processor.tokenizer
    col = QwenAsrCollator(real_processor, loader=loader_of({"a": 1.0, "b": 3.2}))
    feats = [{"audio_path": "a", "text": "আমি যাব"}, {"audio_path": "b", "text": "তুমি এসো কাল সকালে"}]
    b = col(feats)
    assert "labels" in b and "input_features" in b and "input_features_mask" in b
    assert tuple(b["labels"].shape) == tuple(b["input_ids"].shape)
    for ids, labels, f in zip(b["input_ids"].tolist(), b["labels"].tolist(), feats):
        kept = [t for t in labels if t != IGNORE]
        assert kept[-1] == tok.eos_token_id
        assert tok.decode(kept[:-1]) == f["text"]
        cut = ids.index(tok.convert_tokens_to_ids(MARKER))
        assert tok.decode(ids[:cut + 1]).endswith(PREFIX)
        assert all(t == IGNORE for t in labels[:cut + 1])
        assert kept == ids[cut + 1:cut + 1 + len(kept)]  # the unmasked span starts right after the prefix


def test_real_generation_inputs_match_chat_template_path(real_processor, tmp_path):
    """Our inference inputs (rendered prompt + processor call) equal the model card's
    apply_chat_template(tokenize=True, continue_final_message=True) on the same audio."""
    import soundfile as sf

    audio = (0.1 * np.sin(np.arange(16000 * 2) / 16000 * 2 * np.pi * 300)).astype("float32")
    wav = tmp_path / "a.wav"
    sf.write(str(wav), audio, 16000)
    real_processor.tokenizer.padding_side = "left"
    ref = real_processor.apply_chat_template([conversation(str(wav))], tokenize=True, return_dict=True,
                                             padding=True, continue_final_message=True)
    ours = generation_inputs(real_processor, prompt_text(real_processor), [audio])
    assert ours["input_ids"].tolist() == ref["input_ids"].tolist()
    assert ours["input_features"].shape == ref["input_features"].shape
