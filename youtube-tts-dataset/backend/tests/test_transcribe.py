from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ytds.pipeline import transcribe as tr
from ytds.pipeline.audio16k import load_16k

DATA = Path(__file__).parent / "data"
BN_CLIP = DATA / "bn_clip.wav"


class FakeGpu:
    def __init__(self, models):
        self.models = models
        self.used: list[str] = []

    def use(self, name):
        self.used.append(name)
        return self.models[name]


def _vocals(tmp_path) -> Path:
    sr = 44100
    wav = tmp_path / "vocals.wav"
    sf.write(str(wav), np.zeros(sr * 12, dtype="float32"), sr, subtype="PCM_16")
    return wav


def test_transcribe_clips_model_a_only(tmp_path):
    calls: list[int] = []

    def model_a(audios):
        calls.append(len(audios))
        return ["আমার সোনার বাংলা", "", "hello world"][: len(audios)]

    gpu = FakeGpu({"model_a": model_a})
    clips = [{"idx": i, "start_s": i * 3.0, "end_s": i * 3.0 + 2.5} for i in range(3)]
    clips.insert(1, {"idx": 9, "start_s": 0.0, "end_s": 1.0, "status": "dropped"})
    progress: list[tuple[int, int]] = []

    tr.transcribe_clips(clips, _vocals(tmp_path), gpu, lambda d, t: progress.append((d, t)))

    assert gpu.used == ["model_a"]
    kept = [c for c in clips if c.get("status") != "dropped"]
    assert [c["text_a"] for c in kept] == ["আমার সোনার বাংলা", "", "hello world"]
    assert [c["status"] for c in kept] == ["auto", "pending_llm", "pending_llm"]
    assert all(c["text_b"] is None and c["cer"] is None and c["text_source"] == "stt" for c in kept)
    dropped = clips[1]
    assert dropped["status"] == "dropped" and "text_a" not in dropped
    assert progress[-1] == (3, 3)

def test_transcribe_clips_no_live_clips_skips_gpu(tmp_path):
    gpu = FakeGpu({})
    clips = [{"idx": 0, "start_s": 0.0, "end_s": 1.0, "status": "dropped"}]
    tr.transcribe_clips(clips, _vocals(tmp_path), gpu, lambda d, t: None)
    assert gpu.used == []


def test_batches_of_eight():
    seen = []

    def model(audios):
        seen.append(len(audios))
        return [""] * len(audios)

    assert tr.in_batches(model, [np.zeros(10)] * 19) == [""] * 19
    assert seen == [8, 8, 3]


def test_ported_qwen_helpers():
    assert tr.split_transcript("language Bengali<asr_text> আমার সোনার ") == "আমার সোনার"
    assert tr.split_transcript("no marker") == "no marker"
    assert tr.max_new_tokens(2.0) == 76


@pytest.mark.gpu
def test_model_a_loads_and_transcribes():
    model = tr.load_model_a()
    out = model([load_16k(BN_CLIP)])
    print("model_a:", out)
    assert len(out) == 1 and any("ঀ" <= ch <= "৿" for ch in out[0])


def test_strip_glosses_removes_braced_english():
    assert tr.strip_glosses("একটা স্কিল {Skill} দুইটা") == "একটা স্কিল দুইটা"
    assert tr.strip_glosses("ইনফরমেশন {information} টা") == "ইনফরমেশন টা"
    assert tr.strip_glosses("কোনো কিছু") == "কোনো কিছু"
