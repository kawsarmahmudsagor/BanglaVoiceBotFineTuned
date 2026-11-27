from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ytds.pipeline import transcribe as tr
from ytds.pipeline.audio16k import load_16k

DATA = Path(__file__).parent / "data"
BN_CLIP = DATA / "bn_clip.wav"


def _vocals(tmp_path) -> Path:
    sr = 44100
    wav = tmp_path / "vocals.wav"
    sf.write(str(wav), np.zeros(sr * 12, dtype="float32"), sr, subtype="PCM_16")
    return wav


def test_transcribe_pieces_a_slices_vocals_and_reports_progress(tmp_path):
    seen: list[list[int]] = []

    def model_a(audios):
        seen.append([len(a) for a in audios])
        return [f"t{i}" for i in range(len(audios))]

    pieces = [{"idx": i, "start_s": i * 1.0, "end_s": i * 1.0 + 0.5} for i in range(10)]
    progress: list[tuple[int, int]] = []
    out = tr.transcribe_pieces_a(pieces, _vocals(tmp_path), model_a, lambda d, t: progress.append((d, t)))

    assert out == [f"t{i}" for i in range(8)] + ["t0", "t1"]
    assert seen == [[8000] * 8, [8000] * 2]  # 0.5 s at 16 kHz, batches of eight
    assert progress == [(8, 10), (10, 10)]


def test_transcribe_pieces_a_strips_glosses_via_model_helpers(tmp_path):
    out = tr.transcribe_pieces_a([{"start_s": 0.0, "end_s": 1.0}], _vocals(tmp_path),
                                 lambda audios: [tr.split_transcript("language Bengali<asr_text>স্কিল {Skill}")],
                                 lambda d, t: None)
    assert out == ["স্কিল"]


def test_transcribe_pieces_a_no_pieces_skips_model(tmp_path):
    def model_a(audios):
        raise AssertionError("not called")

    assert tr.transcribe_pieces_a([], tmp_path / "missing.wav", model_a, lambda d, t: None) == []


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
    from ytds.config import load_settings

    model = tr.load_model_a(load_settings().model_a_path)
    out = model([load_16k(BN_CLIP)])
    print("model_a:", out)
    assert len(out) == 1 and any("ঀ" <= ch <= "৿" for ch in out[0])


def test_strip_glosses_removes_braced_english():
    assert tr.strip_glosses("একটা স্কিল {Skill} দুইটা") == "একটা স্কিল দুইটা"
    assert tr.strip_glosses("ইনফরমেশন {information} টা") == "ইনফরমেশন টা"
    assert tr.strip_glosses("কোনো কিছু") == "কোনো কিছু"
