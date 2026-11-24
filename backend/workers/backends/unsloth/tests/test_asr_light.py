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
                           "bn_normalize_version": BN_NORMALIZE_VERSION, "max_label_tokens": 448,
                           "storage": "audio-ref"}
    other = whisper_spec(cfg.model_copy(update={"base_model": "mozilla-ai/whisper-large-v3-turbo-bn"}))
    assert other.hash() != spec.hash()


def test_whisper_module_import_is_light():
    import app.core.backend_kit.whisper  # noqa: F401
    assert "transformers" not in sys.modules and "datasets" not in sys.modules


def _ex(row, *, probed=1.0, probe=None, tokens=5):
    from app.core.backend_kit.whisper import row_to_example
    seen = {}

    def tokenize(text):
        seen["text"] = text
        return list(range(tokens))

    return row_to_example(row, probe=probe or (lambda p: probed), tokenize=tokenize), seen


def test_row_to_example_ok_stores_path_not_audio():
    out, seen = _ex({"sample_id": "s", "audio_path": "a.wav", "text": "Hello, World!", "duration_s": 2.5})
    assert out == {"sample_id": "s", "audio_path": "a.wav", "duration_s": 2.5, "labels": [0, 1, 2, 3, 4]}
    assert seen["text"] == "hello world"


def test_row_to_example_uses_probe_duration_when_row_has_none():
    out, _ = _ex({"audio_path": "a.wav", "text": "x"}, probed=3.25)
    assert out["duration_s"] == 3.25


def test_row_to_example_skips_long_audio():
    assert _ex({"audio_path": "a.wav", "text": "x"}, probed=35.0)[0] == "duration 35.0 s > 30 s"
    assert _ex({"audio_path": "a.wav", "text": "x", "duration_s": 35.0})[0] == "duration 35.0 s > 30 s"


def test_row_to_example_allows_exactly_30s():
    assert isinstance(_ex({"audio_path": "a.wav", "text": "x"}, probed=30.0)[0], dict)


def test_row_to_example_skips_too_many_label_tokens():
    assert _ex({"audio_path": "a.wav", "text": "x"}, tokens=449)[0] == "449 label tokens > 448"
    assert isinstance(_ex({"audio_path": "a.wav", "text": "x"}, tokens=448)[0], dict)


def test_row_to_example_undecodable_audio_even_with_duration():
    def boom(path):
        raise RuntimeError("bad header")

    out, _ = _ex({"audio_path": "a.wav", "text": "x", "duration_s": 1.0}, probe=boom)
    assert out == "undecodable audio: RuntimeError"


def test_probe_duration_reads_header_and_rejects_garbage(tmp_path, tiny_audio_ds):
    from app.core.backend_kit.whisper import probe_duration
    root, names = tiny_audio_ds()
    assert abs(probe_duration(root / "files" / names[0]) - 0.5) < 1e-3
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    with pytest.raises(Exception):
        probe_duration(bad)


def test_pad_labels_strips_decoder_start_and_pads():
    from app.core.backend_kit.whisper import pad_labels
    assert pad_labels([[9, 1, 2], [9, 3]], start_id=9) == [[1, 2], [3, -100]]
    assert pad_labels([[9, 1], [8, 3, 4]], start_id=9) == [[9, 1, -100], [8, 3, 4]]  # only strip when all start


def test_batch_parts_loads_paths_and_pads():
    from app.core.backend_kit.whisper import batch_parts
    seen = []
    audios, labels = batch_parts([{"audio_path": "x.wav", "labels": [9, 1, 2]},
                                  {"audio_path": "y.wav", "labels": [9, 3]}], 9,
                                 loader=lambda p: seen.append(p) or [0.0])
    assert seen == ["x.wav", "y.wav"] and audios == [[0.0], [0.0]]
    assert labels == [[1, 2], [3, -100]]


def test_build_whisper_assembles_split_results(monkeypatch, tmp_path):
    from app.core.backend_kit import whisper as w
    from app.core.prepare import Skip

    class Tok:
        def set_prefix_tokens(self, **kw):
            pass

    class Proc:
        tokenizer = Tok()

    monkeypatch.setattr(w, "_write_split", lambda items, out, probe, tok: (
        len(items) - 1, [Skip(sample_id="b", split=out.name, reason="r")]))
    res = w.build_whisper(Proc())({"train": [{}, {}], "val": [{}, {}, {}]}, tmp_path)
    assert res.counts == {"train": 1, "val": 2} and res.files == ["train", "val"]
    assert [s.split for s in res.skipped] == ["train", "val"]


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
    assert a["max_steps"] == 20 and a["dataloader_num_workers"] >= 1 and a["remove_unused_columns"] is False and a["label_names"] == ["labels"]
