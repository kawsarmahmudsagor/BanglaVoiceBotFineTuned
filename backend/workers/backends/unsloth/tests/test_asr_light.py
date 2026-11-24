import sys

import pytest

from app.core.contract import read_config
from app.core.text import BN_NORMALIZE_VERSION

WHISPER = "openai/whisper-large-v3-turbo"


def test_whisper_spec_includes_normalizer_version(run_dir):
    from app.core.backend_kit.whisper import whisper_spec
    cfg = read_config(run_dir("asr", WHISPER))
    spec = whisper_spec(cfg)
    assert spec.family == "whisper" and spec.version == 1
    assert spec.params == {"base_model": WHISPER, "sample_rate": 16000,
                           "bn_normalize_version": BN_NORMALIZE_VERSION, "max_label_tokens": 448}
    other = whisper_spec(cfg.model_copy(update={"base_model": "mozilla-ai/whisper-large-v3-turbo-bn"}))
    assert other.hash() != spec.hash()


def test_whisper_module_import_is_light():
    import app.core.backend_kit.whisper  # noqa: F401
    assert "transformers" not in sys.modules and "datasets" not in sys.modules


def _ex(row, *, audio_len=16000, load=None, tokens=5):
    from app.core.backend_kit.whisper import row_to_example
    seen = {}

    def tokenize(text):
        seen["text"] = text
        return list(range(tokens))

    out = row_to_example(row, load=load or (lambda p: [0.0] * audio_len), extract=lambda a: ("feat", len(a)),
                         tokenize=tokenize)
    return out, seen


def test_row_to_example_ok_normalizes_text():
    out, seen = _ex({"audio_path": "a.wav", "text": "Hello, World!"})
    assert out == {"input_features": ("feat", 16000), "labels": [0, 1, 2, 3, 4]}
    assert seen["text"] == "hello world"


def test_row_to_example_skips_long_audio():
    out, _ = _ex({"audio_path": "a.wav", "text": "x"}, audio_len=35 * 16000)
    assert out == "duration 35.0 s > 30 s"


def test_row_to_example_allows_exactly_30s():
    out, _ = _ex({"audio_path": "a.wav", "text": "x"}, audio_len=30 * 16000)
    assert isinstance(out, dict)


def test_row_to_example_skips_too_many_label_tokens():
    out, _ = _ex({"audio_path": "a.wav", "text": "x"}, tokens=449)
    assert out == "449 label tokens > 448"
    assert isinstance(_ex({"audio_path": "a.wav", "text": "x"}, tokens=448)[0], dict)


def test_row_to_example_undecodable_audio():
    def boom(path):
        raise RuntimeError("bad header")

    out, _ = _ex({"audio_path": "a.wav", "text": "x"}, load=boom)
    assert out == "undecodable audio: RuntimeError"


def test_pad_labels_strips_decoder_start_and_pads():
    from app.core.backend_kit.whisper import pad_labels
    assert pad_labels([[9, 1, 2], [9, 3]], start_id=9) == [[1, 2], [3, -100]]
    assert pad_labels([[9, 1], [8, 3, 4]], start_id=9) == [[9, 1, -100], [8, 3, 4]]  # only strip when all start


def test_build_whisper_streams_and_records_skips(monkeypatch, tmp_path):
    """build_whisper delegates per-row work to row_to_example and reports skips; dataset writing is stubbed."""
    from app.core.backend_kit import whisper as w

    written = {}

    def fake_write(gen_rows, out_dir, n_mels, frames):
        written[out_dir.name] = list(gen_rows)  # consumed lazily as an iterator

    monkeypatch.setattr(w, "_write_split", fake_write)

    class FakeProc:
        class feature_extractor:
            feature_size = 128
            nb_max_frames = 3000

    def fake_row(row, **kw):
        return "too long" if row["sample_id"] == "b" else {"input_features": 1, "labels": [1]}

    monkeypatch.setattr(w, "row_to_example", fake_row)
    monkeypatch.setattr(w, "_make_fns", lambda processor, sr: (None, None, None))
    rows = {"train": [{"sample_id": "a"}, {"sample_id": "b"}], "val": [{"sample_id": "c"}]}
    res = w.build_whisper(FakeProc())(rows, tmp_path)
    assert res.counts == {"train": 1, "val": 1}
    assert [(s.sample_id, s.split, s.reason) for s in res.skipped] == [("b", "train", "too long")]
    assert res.files == ["train", "val"]
    assert len(written["train"]) == 1


def test_asr_handlers_and_export_formats(run_dir):
    from mlops_backend import asr
    assert asr.train is asr.train_asr and asr.eval is asr.eval_asr and callable(asr.prepare)
    rd = run_dir("asr", WHISPER)
    assert set(asr.asr_exporters(rd, read_config(rd))) == {"merged-safetensors", "ct2"}


def test_ct2_command_uses_venv_python(tmp_path):
    from mlops_backend.asr import ct2_command
    cmd = ct2_command(tmp_path / "m", tmp_path / "o")
    assert cmd[0] == sys.executable or cmd[0].endswith("ct2-transformers-converter")
    assert cmd[cmd.index("--quantization") + 1] == "float16"
    assert cmd[cmd.index("--copy_files") + 1:cmd.index("--copy_files") + 3] == ["tokenizer.json", "preprocessor_config.json"]
    assert cmd[cmd.index("--model") + 1] == str(tmp_path / "m")


def test_asr_exporter_requires_complete_final(run_dir):
    from mlops_backend.asr import asr_exporters
    rd = run_dir("asr", WHISPER)
    (rd.checkpoints_dir / "final").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no completed final checkpoint"):
        asr_exporters(rd, read_config(rd))["ct2"](rd.export_dir / "ct2")


def test_eval_asr_requires_complete_final(run_dir):
    from mlops_backend.asr import eval_asr
    rd = run_dir("asr", WHISPER)
    (rd.checkpoints_dir / "final").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no completed final checkpoint"):
        eval_asr(rd, read_config(rd), None)


def test_asr_baseline_eval_key_varies(run_dir):
    from app.core.contract import InitRef
    from mlops_backend.asr import _asr_baseline_key
    cfg = read_config(run_dir("asr", WHISPER))
    a = _asr_baseline_key(cfg, None)
    assert _asr_baseline_key(cfg, 5) != a
    w = cfg.model_copy(update={"init_from": InitRef(path="p", source_run="r0", checkpoint="final",
                                                    method="lora", format="hf-peft")})
    assert _asr_baseline_key(w, None)["init"] == ["r0", "final"]


def test_asr_trainer_args(run_dir):
    from mlops_backend.asr import asr_trainer_args
    rd = run_dir("asr", WHISPER, {"max_steps": 20})
    rd.prepared_refs_path.write_text('[{"role": "train", "path": "/p", "counts": {"train": 8, "val": 2}}]')
    a = asr_trainer_args(read_config(rd), rd)
    assert a["max_steps"] == 20 and a["remove_unused_columns"] is False and a["label_names"] == ["labels"]
