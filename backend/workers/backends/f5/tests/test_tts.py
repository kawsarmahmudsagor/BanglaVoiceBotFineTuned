"""F5 train / eval / export (light: no torch / f5_tts / network, except the heavy-marked ones)."""
import json
import math

import pytest

from app.core.backend_kit import NanGuard, StopFlag, TrainingDiverged, latest_checkpoint, read_init_meta
from app.core.contract import Phase, read_config, read_metrics

from mlops_backend.tts import (
    TEXT_EMBED_KEY, TrainFinished, TrainStopped, UpdateHook, ema_export_state, pick_state_dict, pick_weights_file,
    plan_expansion, publish_checkpoint, save_interval, strip_to_model_keys, total_updates, update_dir,
    updates_per_epoch, check_load_result, pick_ref,
)

MODEL = "ai4bharat/IndicF5"


# ---- key mapping ------------------------------------------------------------------------------------------------

def test_strip_wrapper_prefix_and_drop_bookkeeping():
    sd = {"ema_model.transformer.text_embed.text_embed.weight": 1, "ema_model.transformer.proj_out.bias": 2,
          "ema_model.initted": 3, "ema_model.step": 4, "vocoder.x": 5}
    assert strip_to_model_keys(sd) == {TEXT_EMBED_KEY: 1, "transformer.proj_out.bias": 2}


def test_strip_prefers_ema_copy_when_both_present():
    sd = {"model.transformer.text_embed.text_embed.weight": "online",
          "ema_model.transformer.text_embed.text_embed.weight": "ema"}
    assert strip_to_model_keys(sd) == {TEXT_EMBED_KEY: "ema"}


def test_strip_no_prefix_is_identity():
    sd = {TEXT_EMBED_KEY: 1, "transformer.x": 2}
    assert strip_to_model_keys(sd) == sd


def test_strip_unrecognized_layout_fails_loudly():
    with pytest.raises(ValueError, match="text_embed"):
        strip_to_model_keys({"foo.bar": 1})


def test_pick_state_dict_variants():
    assert pick_state_dict({"ema_model_state_dict": {"a": 1}, "model_state_dict": {"a": 2}}) == {"a": 1}
    assert pick_state_dict({"ema_model_state_dict": {"a": 1}, "model_state_dict": {"a": 2}}, ema=False) == {"a": 2}
    assert pick_state_dict({"model_state_dict": {"a": 2}}) == {"a": 2}
    assert pick_state_dict({"a": 3}) == {"a": 3}


def test_ema_export_state_strips_ema_prefix():
    ck = {"ema_model_state_dict": {"ema_model.transformer.text_embed.text_embed.weight": 1, "initted": 0, "step": 5},
          "model_state_dict": {}, "update": 7}
    assert ema_export_state(ck) == {TEXT_EMBED_KEY: 1}


def test_pick_weights_file():
    assert pick_weights_file(["config.json", "model.safetensors", "vocab.txt"]) == "model.safetensors"
    assert pick_weights_file(["a/model.safetensors", "model.safetensors"]) == "model.safetensors"
    assert pick_weights_file(["README.md", "checkpoints/model_1250000.pt", "vocab.txt"]) == "checkpoints/model_1250000.pt"
    assert pick_weights_file(["x.json", "sub/deep/w.safetensors"]) == "sub/deep/w.safetensors"
    with pytest.raises(FileNotFoundError, match="weights"):
        pick_weights_file(["config.json", "vocab.txt"])


def test_plan_expansion():
    assert plan_expansion(rows=11, vocab_size=10) == 0  # rows = vocab + 1 (index 0 is the filler)
    assert plan_expansion(rows=11, vocab_size=15) == 5
    with pytest.raises(ValueError, match="smaller"):
        plan_expansion(rows=11, vocab_size=8)


def test_check_load_result_ignores_mel_buffers_only():
    check_load_result(["mel_spec.mel_stft.mel_scale.fb"], ["mel_spec.mel_stft.spectrogram.window"])
    with pytest.raises(RuntimeError, match="transformer.a") as e:
        check_load_result(["transformer.a"], ["wrapper.b"])
    assert "wrapper.b" in str(e.value)


@pytest.mark.heavy
def test_embedding_expansion_shape():
    import torch

    from mlops_backend.tts import expand_rows

    w = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    out = expand_rows(w, 2)
    assert out.shape == (6, 3) and torch.equal(out[:4], w)
    assert torch.allclose(out[4], w.mean(0)) and torch.allclose(out[5], w.mean(0))
    assert expand_rows(w, 0).shape == (4, 3)


# ---- update dirs / checkpoint contract ---------------------------------------------------------------------------

def _cfg(run_dir, **hp):
    rd = run_dir("tts", MODEL, hp)
    return rd, read_config(rd)


def test_update_dir_naming_and_latest(run_dir, tmp_path):
    rd, cfg = _cfg(run_dir)
    src = tmp_path / "model_last.pt"
    src.write_bytes(b"w")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("a\nb\n", encoding="utf-8")
    assert update_dir(rd, 40).name == "update-40"
    for n in (9, 100, 40):
        publish_checkpoint(src, update_dir(rd, n), cfg, n, vocab)
    (rd.checkpoints_dir / "update-500").mkdir()  # no .complete: not a checkpoint
    assert latest_checkpoint(rd) == update_dir(rd, 100)


def test_publish_checkpoint_meta_before_complete_and_contents(run_dir, tmp_path):
    rd, cfg = _cfg(run_dir)
    src = tmp_path / "model_last.pt"
    src.write_bytes(b"weights")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("a\n", encoding="utf-8")
    d = update_dir(rd, 3)
    seen = []
    orig = __import__("mlops_backend.tts", fromlist=["mark_complete"]).mark_complete

    def spy(p):
        seen.append((p / "init_meta.json").is_file())
        orig(p)

    import mlops_backend.tts as t

    t.mark_complete = spy
    try:
        publish_checkpoint(src, d, cfg, 3, vocab)
    finally:
        t.mark_complete = orig
    assert seen == [True]
    assert (d / "model_last.pt").read_bytes() == b"weights" and (d / "vocab.txt").read_text(encoding="utf-8") == "a\n"
    meta = read_init_meta(d)
    assert meta["format"] == "f5-pt" and meta["backend"] == "f5" and meta["method"] == "full" and meta["step"] == 3
    assert (d / ".complete").exists() and not list(d.glob("*.tmp"))


def test_publish_checkpoint_resave_overwrites(run_dir, tmp_path):
    rd, cfg = _cfg(run_dir)
    src = tmp_path / "m.pt"
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("a\n", encoding="utf-8")
    d = update_dir(rd, 1)
    src.write_bytes(b"1")
    publish_checkpoint(src, d, cfg, 1, vocab)
    src.write_bytes(b"22")
    publish_checkpoint(src, d, cfg, 1, vocab)
    assert (d / "model_last.pt").read_bytes() == b"22" and (d / ".complete").exists()


# ---- schedule helpers --------------------------------------------------------------------------------------------

def test_updates_per_epoch_and_total():
    # 100 clips x 2.56 s = 256 s -> 256*24000/256 = 24000 frames; batch 3200 frames -> 8 batches; accum 2 -> 4 updates
    assert updates_per_epoch([2.56] * 100, 3200, 2) == 4
    assert updates_per_epoch([], 3200, 1) == 1
    assert total_updates({"epochs": 3}, 4) == 12
    assert total_updates({"epochs": 2.5}, 4) == 10
    assert total_updates({"epochs": 3, "max_steps": 20}, 4) == 20
    assert total_updates({}, 4) == 4


def test_save_interval():
    assert save_interval({"save_steps": 7}, 1000) == 7
    assert save_interval({}, 1000) == 100
    assert save_interval({}, 100) == 50


# ---- per-update hook ---------------------------------------------------------------------------------------------

def _hook(run_dir, total=10, saved=None):
    from app.core.backend_kit import Reporter

    rd = run_dir("tts", MODEL)
    stop = StopFlag()
    saved = [] if saved is None else saved
    return rd, stop, saved, UpdateHook(Reporter(rd), stop, NanGuard(), total, saved.append)


def test_hook_logs_metrics_and_dedupes_repeated_step(run_dir):
    rd, stop, saved, hook = _hook(run_dir)
    hook({"loss": 0.5, "lr": 1e-5}, step=1)
    hook({"loss": 0.5, "lr": 1e-5}, step=1)  # accumulation micro-batch repeats the step
    hook({"loss": 0.4, "lr": 1e-5}, step=2)
    ms = [(m.step, m.name, m.value) for m in read_metrics(rd)[0]]
    assert ms.count((1, "loss", 0.5)) == 1 and (2, "loss", 0.4) in ms and (1, "lr", 1e-5) in ms
    assert saved == []


def test_hook_nan_guard_after_three(run_dir):
    _, _, _, hook = _hook(run_dir)
    hook({"loss": math.nan}, step=1)
    hook({"loss": math.nan}, step=2)
    with pytest.raises(TrainingDiverged, match="lower learning_rate"):
        hook({"loss": math.nan}, step=3)


def test_hook_stop_saves_then_stops(run_dir):
    _, stop, saved, hook = _hook(run_dir)
    hook({"loss": 1.0}, step=1)
    stop.requested = True
    with pytest.raises(TrainStopped):
        hook({"loss": 1.0}, step=2)
    assert saved == [2]


def test_hook_finishes_at_total(run_dir):
    _, _, saved, hook = _hook(run_dir, total=3)
    hook({"loss": 1.0}, step=2)
    with pytest.raises(TrainFinished):
        hook({"loss": 1.0}, step=3)
    assert saved == [3]


def test_hook_stop_at_natural_end_counts_as_finished(run_dir):
    _, stop, saved, hook = _hook(run_dir, total=3)
    stop.requested = True
    with pytest.raises(TrainFinished):
        hook({"loss": 1.0}, step=3)


def test_hook_status_is_train_phase(run_dir):
    from app.core.contract import read_status

    rd, _, _, hook = _hook(run_dir, total=10)
    hook({"loss": 1.0}, step=4)
    st = read_status(rd)
    assert st.phase == Phase.train and st.step == 4 and st.total_steps == 10


# ---- reference choice + dispatch ---------------------------------------------------------------------------------

def test_pick_ref_speaker_then_default_then_first(tmp_path):
    for n in ("default", "a", "b c"):
        (tmp_path / f"{n.replace(' ', '_')}.wav").write_bytes(b"")
        (tmp_path / f"{n.replace(' ', '_')}.txt").write_text("t", encoding="utf-8")
    assert pick_ref(tmp_path, "a").name == "a.wav"
    assert pick_ref(tmp_path, "b c").name == "b_c.wav"
    assert pick_ref(tmp_path, "zzz").name == "default.wav"
    (tmp_path / "default.wav").unlink()
    assert pick_ref(tmp_path, None).name == "a.wav"
    for f in tmp_path.glob("*.wav"):
        f.unlink()
    with pytest.raises(FileNotFoundError):
        pick_ref(tmp_path, "a")


def test_family_module_exposes_phases():
    from mlops_backend import f5

    assert callable(f5.prepare) and callable(f5.train) and callable(f5.eval) and callable(f5.export)


def test_eval_requires_complete_final(run_dir):
    from mlops_backend.tts import eval_f5

    rd, cfg = _cfg(run_dir)
    (rd.checkpoints_dir / "final").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no completed final checkpoint"):
        eval_f5(rd, cfg, None)


def test_export_requires_complete_final_and_names_format(run_dir):
    from mlops_backend.tts import f5_exporters

    rd, cfg = _cfg(run_dir)
    ex = f5_exporters(rd, cfg)
    assert set(ex) == {"f5-ckpt"}
    with pytest.raises(RuntimeError, match="no completed final checkpoint"):
        ex["f5-ckpt"](rd.export_dir / "f5-ckpt")


def test_baseline_key_has_max_samples_and_init(run_dir):
    from app.core.contract import InitRef
    from mlops_backend.tts import f5_baseline_key

    rd = run_dir("tts", MODEL)
    cfg = read_config(rd)
    assert f5_baseline_key(cfg, 5)["max_samples"] == 5
    cfg2 = cfg.model_copy(update={"init_from": InitRef(path="/x", source_run="r0", checkpoint="final", method="full", format="f5-pt")})
    assert f5_baseline_key(cfg2, 5) != f5_baseline_key(cfg, 5)


# ---- heavy (GPU VM only) -----------------------------------------------------------------------------------------

@pytest.mark.heavy
def test_f5_20_updates_and_synthesize(run_dir, monkeypatch, tmp_path):
    from app.core.backend_kit import Reporter
    from app.core.contract import read_eval_results
    from mlops_backend import f5 as p

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    rd = run_dir("tts", MODEL, {"max_steps": 20, "batch_size_frames": 3200, "grad_accum": 1, "learning_rate": 1e-5,
                                "warmup_steps": 2, "save_steps": 10, "eval": {"max_samples": 2}},
                 export=["f5-ckpt"])
    cfg = read_config(rd)
    rep = Reporter(rd)
    assert p.prepare(rd, cfg, rep) == 0
    assert p.train(rd, cfg, rep) == 0
    final = rd.checkpoints_dir / "final"
    assert (final / ".complete").exists() and (final / "model_last.pt").is_file() and (final / "vocab.txt").is_file()
    assert read_init_meta(final)["format"] == "f5-pt"
    assert (update_dir(rd, 10) / ".complete").exists() and (update_dir(rd, 20) / ".complete").exists()
    assert p.eval(rd, cfg, rep) == 0
    res = read_eval_results(rd).metrics
    assert "rtf" in res
    assert p.export(rd, cfg, rep) == 0
    out = rd.export_dir / "f5-ckpt"
    assert (out / "model.safetensors").is_file() and (out / "vocab.txt").is_file() and (out / "refs").is_dir()
    assert json.loads(rd.export_manifest_path.read_text())["artifacts"][0]["serve_with"] == "f5-tts"


@pytest.mark.heavy
def test_torchaudio_reads_source_formats(tmp_path):
    """F5's CustomDataset loads audio_path with torchaudio: it must open the formats users bring."""
    import subprocess

    import numpy as np
    import torchaudio

    from app.core.backend_kit.audio import write_flac, write_wav

    t = np.arange(24000) / 24000
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    write_wav(tmp_path / "a.wav", audio, 24000)
    write_flac(tmp_path / "a.flac", audio, 24000)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_path / "a.wav"), str(tmp_path / "a.mp3")],
                   check=True)
    for name in ("a.wav", "a.flac", "a.mp3"):
        wav, sr = torchaudio.load(str(tmp_path / name))
        assert wav.shape[0] >= 1 and wav.shape[1] > 20000 and sr == 24000
