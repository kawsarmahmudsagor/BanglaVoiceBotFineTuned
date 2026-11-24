"""CosyVoice train / eval / export (light: no torch / cosyvoice / network, except the heavy-marked test)."""
import json

import pytest

from app.core.backend_kit import read_init_meta
from app.core.contract import read_config

from mlops_backend.export import link_or_copy
from mlops_backend.eval import split_sentences
from mlops_backend.train import (
    CKPT_FORMAT, COMPONENT_FORMAT, assemble_final, checkpoint_dirs, clean_state, component_sources, epoch_dir,
    keep_count, latest_complete, plan_components, prune_checkpoints, publish_checkpoint, resolve_components,
)

MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"


def _cfg(run_dir, **hp):
    rd = run_dir("tts", MODEL, hp)
    return rd, read_config(rd)


# ---- component planning ------------------------------------------------------------------------------------------

def test_component_order_and_resume_skip():
    assert plan_components(["hifigan", "llm", "flow"], []) == ["llm", "flow", "hifigan"]
    assert plan_components(["flow", "llm"], []) == ["llm", "flow"]
    assert plan_components(["llm", "flow", "hifigan"], ["llm"]) == ["flow", "hifigan"]
    assert plan_components(["llm", "flow"], ["llm", "flow"]) == []
    assert plan_components(["flow"], ["llm"]) == ["flow"]  # completed but not requested: ignored
    with pytest.raises(ValueError, match="unknown component"):
        plan_components(["llm", "vocoder"], [])


def test_resolve_components_default_and_requested():
    assert resolve_components({}) == ["llm"]
    assert resolve_components({"components": ["hifigan", "llm"]}) == ["llm", "hifigan"]


# ---- checkpoint dirs / retention ---------------------------------------------------------------------------------

def _mk(rd, comp, n, step=None, complete=True):
    d = epoch_dir(rd, comp, n, step)
    d.mkdir(parents=True)
    (d / "model.pt").write_bytes(b"x")
    if complete:
        (d / ".complete").write_text("")
    return d


def test_epoch_dir_naming(run_dir):
    rd, _ = _cfg(run_dir)
    assert epoch_dir(rd, "llm", 3).name == "llm-epoch-3"
    assert epoch_dir(rd, "llm", 3, 120).name == "llm-epoch-3-step-120"


def test_latest_complete_orders_step_saves_before_epoch_end(run_dir):
    rd, _ = _cfg(run_dir)
    _mk(rd, "llm", 1)
    _mk(rd, "llm", 2, step=50)   # mid second epoch
    assert latest_complete(rd, "llm") == epoch_dir(rd, "llm", 2, 50)
    _mk(rd, "llm", 2)            # second epoch finished
    assert latest_complete(rd, "llm") == epoch_dir(rd, "llm", 2)
    _mk(rd, "llm", 9, complete=False)
    _mk(rd, "flow", 5)
    assert latest_complete(rd, "llm") == epoch_dir(rd, "llm", 2)
    assert latest_complete(rd, "hifigan") is None
    assert [d.name for _, d, _ in checkpoint_dirs(rd, "flow")] == ["flow-epoch-5"]


def test_prune_keeps_latest_k_per_component_and_spares_final(run_dir):
    rd, _ = _cfg(run_dir)
    for n in (1, 2, 3, 4):
        _mk(rd, "llm", n)
    _mk(rd, "llm", 2, step=10)  # sorts before llm-epoch-2: removable
    _mk(rd, "llm", 5, complete=False)  # newer than the newest complete: in progress, kept
    for n in (1, 2, 3):
        _mk(rd, "flow", n)
    (rd.checkpoints_dir / "llm-final").mkdir()
    (rd.checkpoints_dir / "final").mkdir()
    (rd.checkpoints_dir / "init").mkdir()
    prune_checkpoints(rd, "llm", 2)
    left = sorted(d.name for d in rd.checkpoints_dir.iterdir())
    assert left == ["final", "flow-epoch-1", "flow-epoch-2", "flow-epoch-3", "init", "llm-epoch-3", "llm-epoch-4",
                    "llm-epoch-5", "llm-final"]


def test_prune_never_deletes_newest_even_with_k_zero(run_dir):
    rd, _ = _cfg(run_dir)
    for n in (1, 2):
        _mk(rd, "llm", n)
    prune_checkpoints(rd, "llm", 0)
    assert [d.name for d in rd.checkpoints_dir.iterdir()] == ["llm-epoch-2"]


def test_keep_count_default_and_override():
    assert keep_count({}) == 2
    assert keep_count({"keep_checkpoints": 5}) == 5
    assert keep_count({"keep_checkpoints": 0}) == 1


def test_publish_checkpoint_meta_before_complete(run_dir, tmp_path, monkeypatch):
    import mlops_backend.train as t

    rd, cfg = _cfg(run_dir)
    src = tmp_path / "save.pt"
    src.write_bytes(b"weights")
    d = epoch_dir(rd, "llm", 2)
    seen = []
    orig = t.mark_complete
    monkeypatch.setattr(t, "mark_complete", lambda p: (seen.append((p / "init_meta.json").is_file()), orig(p)))
    publish_checkpoint(src, d, cfg, 40)
    assert seen == [True]
    assert not src.exists() and (d / "model.pt").read_bytes() == b"weights" and (d / ".complete").exists()
    meta = read_init_meta(d)
    # a per-component dir is not a warm-start source: resolve_init rejects this format (only `final` is cosyvoice-pt)
    assert meta["format"] == COMPONENT_FORMAT == "cosyvoice-component" and CKPT_FORMAT == "cosyvoice-pt"
    assert meta["backend"] == "cosyvoice" and meta["step"] == 40


# ---- state dict cleaning -----------------------------------------------------------------------------------------

def test_clean_state_drops_bookkeeping():
    sd = {"a": 1, "b": 2, "epoch": 3, "step": 40}
    assert clean_state(sd, "llm") == {"a": 1, "b": 2}
    assert clean_state(sd, "flow") == {"a": 1, "b": 2}


def test_clean_state_hifigan_keeps_generator_only_without_prefix():
    sd = {"generator.conv.w": 1, "generator.f0.b": 2, "discriminator.mpd.w": 3, "epoch": 1, "step": 5}
    assert clean_state(sd, "hifigan") == {"conv.w": 1, "f0.b": 2}
    base = {"conv.w": 1}  # a plain hift.pt is already generator-only
    assert clean_state(base, "hifigan") == base


# ---- final dir ---------------------------------------------------------------------------------------------------

def _fake_base(root):
    root.mkdir(parents=True)
    for name in ("llm.pt", "flow.pt", "hift.pt", "speech_tokenizer_v3.onnx", "campplus.onnx", "cosyvoice3.yaml"):
        (root / name).write_bytes(f"base-{name}".encode())
    (root / "CosyVoice-BlankEN").mkdir()
    (root / "CosyVoice-BlankEN" / "tokenizer.json").write_text("{}")
    return root


def test_final_dir_fills_untrained_from_base(run_dir, tmp_path):
    rd, cfg = _cfg(run_dir)
    base = _fake_base(tmp_path / "snap")
    trained = tmp_path / "llm-trained.pt"
    trained.write_bytes(b"trained-llm")
    sources = component_sources(rd, cfg, base, {"llm": trained})
    assert sources["llm.pt"] == trained
    assert sources["flow.pt"] == base / "flow.pt" and sources["hift.pt"] == base / "hift.pt"
    final = rd.checkpoints_dir / "final"
    seen = []
    import mlops_backend.train as t
    orig = t.mark_complete
    t.mark_complete = lambda p: (seen.append((p / "init_meta.json").is_file()), orig(p))
    try:
        assemble_final(final, sources, base, cfg, step=7)
    finally:
        t.mark_complete = orig
    assert seen == [True]
    assert (final / "llm.pt").read_bytes() == b"trained-llm"
    assert (final / "flow.pt").read_bytes() == b"base-flow.pt"
    assert (final / "hift.pt").read_bytes() == b"base-hift.pt"
    for name in ("speech_tokenizer_v3.onnx", "campplus.onnx", "cosyvoice3.yaml"):
        assert (final / name).read_bytes() == f"base-{name}".encode()
    assert (final / "CosyVoice-BlankEN" / "tokenizer.json").is_file()
    assert (final / ".complete").exists()
    meta = read_init_meta(final)
    assert meta["format"] == "cosyvoice-pt" and meta["step"] == 7


def test_final_dir_rebuild_replaces_files_without_touching_sources(run_dir, tmp_path):
    rd, cfg = _cfg(run_dir)
    base = _fake_base(tmp_path / "snap")
    final = rd.checkpoints_dir / "final"
    assemble_final(final, component_sources(rd, cfg, base, {}), base, cfg, step=0)
    new = tmp_path / "new-llm.pt"
    new.write_bytes(b"new")
    assemble_final(final, component_sources(rd, cfg, base, {"llm": new}), base, cfg, step=1)
    assert (final / "llm.pt").read_bytes() == b"new"
    assert (base / "llm.pt").read_bytes() == b"base-llm.pt"  # the cache copy is intact


def test_component_sources_prefer_init_dir_over_base(run_dir, tmp_path):
    rd, cfg = _cfg(run_dir)
    base = _fake_base(tmp_path / "snap")
    rd.init_dir.mkdir(parents=True, exist_ok=True)
    (rd.init_dir / "flow.pt").write_bytes(b"init-flow")
    # init_from unset in the fixture: the init dir is ignored
    assert component_sources(rd, cfg, base, {})["flow.pt"] == base / "flow.pt"
    from app.core.contract import InitRef
    cfg2 = cfg.model_copy(update={"init_from": InitRef(path="x", source_run="r0", checkpoint="final",
                                                        method="full", format="cosyvoice-pt")})
    s = component_sources(rd, cfg2, base, {})
    assert s["flow.pt"] == rd.init_dir / "flow.pt" and s["llm.pt"] == base / "llm.pt"


def test_link_or_copy_hardlinks_same_filesystem(tmp_path):
    src = tmp_path / "a.bin"
    src.write_bytes(b"data")
    dst = tmp_path / "sub" / "b.bin"
    link_or_copy(src, dst)
    assert dst.read_bytes() == b"data"
    assert src.stat().st_ino == dst.stat().st_ino or src.stat().st_nlink == 1  # link where the FS supports it


def test_link_or_copy_follows_hf_style_symlink(tmp_path):
    blobs, snap = tmp_path / "blobs", tmp_path / "snapshots" / "rev"
    blobs.mkdir()
    snap.mkdir(parents=True)
    (blobs / "abc123").write_bytes(b"blob")
    try:
        (snap / "llm.pt").symlink_to("../../blobs/abc123")  # relative, like the HF cache
    except OSError:
        pytest.skip("symlinks not permitted")
    dst = tmp_path / "out" / "llm.pt"
    link_or_copy(snap / "llm.pt", dst)
    assert dst.is_file() and not dst.is_symlink() and dst.read_bytes() == b"blob"
    (snap / "llm.pt").unlink()
    assert dst.read_bytes() == b"blob"  # survives removing the symlink


# ---- deepspeed stub ----------------------------------------------------------------------------------------------

def _drop_deepspeed(monkeypatch):
    import sys

    for k in [k for k in sys.modules if k == "deepspeed" or k.startswith("deepspeed.")]:
        monkeypatch.delitem(sys.modules, k)


def test_deepspeed_stub_provides_import_time_names_and_raises(monkeypatch):
    import sys

    from mlops_backend.compat import ensure_deepspeed_stub

    _drop_deepspeed(monkeypatch)
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a, **k: None)
    ensure_deepspeed_stub()
    from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live

    import deepspeed

    for fn in (estimate_zero2_model_states_mem_needs_all_live, deepspeed.init_distributed, deepspeed.initialize,
               deepspeed.add_config_arguments):
        with pytest.raises(RuntimeError, match="deepspeed engine not supported"):
            fn()
    assert deepspeed.runtime.zero.stage_1_and_2 is sys.modules["deepspeed.runtime.zero.stage_1_and_2"]
    ensure_deepspeed_stub()  # idempotent
    for k in [k for k in sys.modules if k == "deepspeed" or k.startswith("deepspeed.")]:
        monkeypatch.delitem(sys.modules, k)


def test_deepspeed_stub_reads_as_not_installed_to_find_spec_checks(monkeypatch):
    """transformers probes optional packages with importlib.util.find_spec, which raises ValueError for a
    sys.modules entry whose __spec__ is None; with a spec it returns, and the metadata lookup says 'not installed'."""
    import importlib.metadata
    import importlib.util
    import sys

    from mlops_backend.compat import ensure_deepspeed_stub

    _drop_deepspeed(monkeypatch)
    if importlib.util.find_spec("deepspeed") is not None:
        pytest.skip("deepspeed is installed here")
    ensure_deepspeed_stub()
    try:
        assert importlib.util.find_spec("deepspeed") is not None
        assert importlib.util.find_spec("deepspeed.runtime.zero.stage_1_and_2") is not None
        with pytest.raises(importlib.metadata.PackageNotFoundError):
            importlib.metadata.version("deepspeed")
    finally:
        for k in [k for k in sys.modules if k == "deepspeed" or k.startswith("deepspeed.")]:
            monkeypatch.delitem(sys.modules, k)


def test_preload_cuda_libs_loads_global_and_retries_dependents(monkeypatch, tmp_path):
    import ctypes
    import sys

    from mlops_backend.compat import preload_cuda_libs

    d = tmp_path / "nvidia" / "cudnn" / "lib"
    d.mkdir(parents=True)
    for n in ("libcudnn_adv.so.9", "libcudnn.so.9", "libbroken.so.1"):
        (d / n).write_bytes(b"")
    loaded, calls = set(), []

    def fake_cdll(path, mode=0):
        name = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        calls.append((name, mode))
        if name == "libbroken.so.1" or (name == "libcudnn_adv.so.9" and "libcudnn.so.9" not in loaded):
            raise OSError(name)
        loaded.add(name)

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    assert preload_cuda_libs([d]) == 2
    assert loaded == {"libcudnn.so.9", "libcudnn_adv.so.9"}
    assert all(mode == ctypes.RTLD_GLOBAL for _, mode in calls)


def test_preload_cuda_libs_is_a_noop_off_linux(monkeypatch, tmp_path):
    import sys

    from mlops_backend.compat import preload_cuda_libs

    monkeypatch.setattr(sys, "platform", "win32")
    assert preload_cuda_libs([tmp_path]) == 0


def test_use_offline_features_unsets_onnx_path(monkeypatch):
    import os

    from mlops_backend.compat import use_offline_features

    monkeypatch.setenv("onnx_path", "/models/cosyvoice3")
    use_offline_features()
    assert "onnx_path" not in os.environ
    use_offline_features()  # idempotent when already unset


def test_train_and_eval_never_set_onnx_path():
    """onnx_path switches upstream to online token extraction (needs an unpublished batch onnx)."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "mlops_backend"
    for name in ("train.py", "eval.py"):
        text = (src / name).read_text(encoding="utf-8")
        assert '"onnx_path": ' not in text and 'environ["onnx_path"]' not in text, name


def test_deepspeed_stub_leaves_a_real_install_alone(monkeypatch):
    import sys
    import types

    from mlops_backend.compat import ensure_deepspeed_stub

    real = types.ModuleType("deepspeed")
    monkeypatch.setitem(sys.modules, "deepspeed", real)
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a, **k: object())
    ensure_deepspeed_stub()
    assert sys.modules["deepspeed"] is real


# ---- epoch loop hooks (fake executor) ----------------------------------------------------------------------------

class _Stop:
    requested = False


def _info(step, batch_idx, epoch=0, loss=0.5):
    return {"tag": "TRAIN", "step": step, "epoch": epoch, "batch_idx": batch_idx, "accum_grad": 1, "lr": 1e-5,
            "loss_dict": {"loss": loss, "acc": 0.9}}


def _fake_world(tmp_path, stop, stop_after_batch=None, n_batches=5):
    import types

    calls = {"cv": 0, "save": []}

    def fake_save(m, name, info):
        calls["save"].append((name, dict(info)))
        (tmp_path / "work" / "save.pt").write_bytes(b"w")

    ex_mod = types.SimpleNamespace(log_per_step=lambda w, i: None, log_per_save=lambda w, i: None,
                                   save_model=fake_save, cosyvoice_join=lambda g, i: False)
    originals = dict(vars(ex_mod))

    class Ex:
        step, epoch = 0, 0

        def cv(self, *a, **k):
            calls["cv"] += 1

        def train_one_epoc(self, model, *a):
            info = None
            for b in range(n_batches):
                if stop_after_batch is not None and b == stop_after_batch:
                    stop.requested = True
                info = _info(self.step, b, self.epoch)
                if ex_mod.cosyvoice_join(None, info):
                    break
                ex_mod.log_per_step(None, info)
                self.step += 1
            self.cv(model, None, None, info)

        train_one_epoc_gan = train_one_epoc

    (tmp_path / "work").mkdir(exist_ok=True)
    return ex_mod, originals, Ex(), calls


def _run(run_dir, tmp_path, stop, ex_mod, ex, max_steps=None, epochs=2):
    from app.core.backend_kit import Reporter

    from mlops_backend.train import _run_epochs

    rd, cfg = _cfg(run_dir)
    info_dict = {"max_epoch": epochs if max_steps is None else 10 ** 6}

    class DS:
        def set_epoch(self, e):
            pass

    work = tmp_path / "work"
    halted = _run_epochs("llm", cfg, rd, Reporter(rd), stop, ex_mod, ex, None, None, None, None, None, DS(), None,
                         None, info_dict, None, work, 0, -1, max_steps, epochs, 1)
    return halted, rd


def test_run_epochs_stop_halts_saves_midepoch_and_restores_names(run_dir, tmp_path):
    stop = _Stop()
    ex_mod, originals, ex, calls = _fake_world(tmp_path, stop, stop_after_batch=2)
    halted, rd = _run(run_dir, tmp_path, stop, ex_mod, ex)
    assert halted == "stop"
    assert ex.step == 2 and calls["cv"] == 0  # join broke the loop at batch 2; the cv wrapper saved, no CV pass
    assert [n for n, _ in calls["save"]] == ["save"]
    assert calls["save"][0][1]["epoch"] == -1 and calls["save"][0][1]["step"] == 2  # resume repeats epoch 0
    assert (epoch_dir(rd, "llm", 1, 2) / ".complete").exists()
    for k, v in originals.items():
        assert getattr(ex_mod, k) is v  # the four names are restored
    assert "cv" not in vars(ex)  # the instance override is removed again


def test_run_epochs_restores_names_when_a_hook_raises(run_dir, tmp_path):
    import math

    from app.core.backend_kit import TrainingDiverged

    stop = _Stop()
    ex_mod, originals, ex, calls = _fake_world(tmp_path, stop)

    def bad(model, *a):
        for b in range(4):
            ex_mod.log_per_step(None, _info(b, b, loss=math.nan))

    ex.train_one_epoc = bad
    with pytest.raises(TrainingDiverged):
        _run(run_dir, tmp_path, stop, ex_mod, ex)
    for k, v in originals.items():
        assert getattr(ex_mod, k) is v


def test_run_epochs_stop_between_epochs_returns_stop(run_dir, tmp_path):
    stop = _Stop()
    ex_mod, originals, ex, calls = _fake_world(tmp_path, stop, n_batches=2)
    orig = ex.train_one_epoc

    def epoch(model, *a):
        orig(model, *a)
        stop.requested = True

    ex.train_one_epoc = epoch
    halted, _ = _run(run_dir, tmp_path, stop, ex_mod, ex, epochs=3)
    assert halted == "stop" and ex.epoch == 0 and calls["cv"] == 1


def test_run_epochs_max_steps_halts_with_max(run_dir, tmp_path):
    stop = _Stop()
    ex_mod, originals, ex, calls = _fake_world(tmp_path, stop)
    halted, rd = _run(run_dir, tmp_path, stop, ex_mod, ex, max_steps=3)
    assert halted == "max" and ex.step == 3
    assert (epoch_dir(rd, "llm", 1, 3) / ".complete").exists()


def test_train_metrics_gan_reports_d_and_g_turns_separately():
    from mlops_backend.train import train_metrics

    seen: dict = {}
    d = train_metrics(_info(4, 0), True, seen)
    g = train_metrics(_info(4, 0), True, seen)
    assert [n for n, _ in d] == ["loss_d"] and sorted(n for n, _ in g) == ["loss_g", "lr"]
    assert [n for n, _ in train_metrics(_info(5, 1), True, seen)] == ["loss_d"]  # new batch: D turn first again
    assert dict(d)["loss_d"] == 0.5


def test_train_metrics_non_gan_and_accum_boundary():
    from mlops_backend.train import train_metrics

    assert dict(train_metrics(_info(0, 0), False, {})) == {"loss": 0.5, "acc": 0.9, "lr": 1e-5}
    info = _info(0, 0)
    info["accum_grad"] = 2
    assert train_metrics(info, False, {}) == []  # not an accumulation boundary


def test_env_restored_after_snapshot(monkeypatch):
    import os

    from mlops_backend.train import restore_env, snapshot_env

    monkeypatch.setenv("RANK", "7")
    monkeypatch.delenv("MASTER_PORT", raising=False)
    snap = snapshot_env()
    os.environ["RANK"] = "0"
    os.environ["MASTER_PORT"] = "1234"
    restore_env(snap)
    assert os.environ["RANK"] == "7" and "MASTER_PORT" not in os.environ


# ---- text --------------------------------------------------------------------------------------------------------

def test_split_sentences_on_danda():
    assert split_sentences("আমি যাব। তুমি এসো। শেষ") == ["আমি যাব।", "তুমি এসো।", "শেষ"]
    assert split_sentences("একটি বাক্য") == ["একটি বাক্য"]
    assert split_sentences("") == []


# ---- heavy (GPU VM only) -----------------------------------------------------------------------------------------

@pytest.mark.heavy
def test_cosyvoice_20_steps_llm_and_synthesize(run_dir, monkeypatch, tmp_path):
    from app.core.backend_kit import Reporter
    from app.core.contract import read_eval_results

    from mlops_backend import cosyvoice as p

    # prepare reads run_dir's own dataset; clips under 1 s are skipped (MIN_SECONDS), so make them 1.5 s
    rd = run_dir("tts", MODEL, {"components": ["llm"], "max_steps": 20, "grad_accum": 1, "learning_rate": 1e-5,
                                "warmup_steps": 0, "save_steps": 10, "keep_checkpoints": 2,
                                "eval": {"max_samples": 2}}, export=["cosyvoice-dir"], seconds=1.5)
    cfg = read_config(rd)
    rep = Reporter(rd)
    assert p.prepare(rd, cfg, rep) == 0
    assert p.train(rd, cfg, rep) == 0
    final = rd.checkpoints_dir / "final"
    assert (final / ".complete").exists() and (final / "llm.pt").is_file() and (final / "flow.pt").is_file()
    assert (final / "cosyvoice3.yaml").is_file() and (final / "CosyVoice-BlankEN").is_dir()
    assert read_init_meta(final)["format"] == "cosyvoice-pt"
    assert p.eval(rd, cfg, rep) == 0
    res = read_eval_results(rd).metrics
    assert "rtf" in res and "eval_loss_llm" in res
    assert p.export(rd, cfg, rep) == 0
    out = rd.export_dir / "cosyvoice-dir"
    assert (out / "llm.pt").is_file() and (out / "refs").is_dir()
    assert json.loads((out / "serve.json").read_text()) == {"serve_with": "cosyvoice"}


def test_download_model_uses_the_default_hf_cache(monkeypatch, tmp_path):
    import sys
    import types

    from mlops_backend.cosyvoice import download_model

    seen = {}
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda repo, **kw: (seen.update(kw), str(tmp_path))[1]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert download_model("a/b") == tmp_path
    assert seen.get("cache_dir") is None


def test_check_finds_snapshot_under_hf_home_hub(monkeypatch, tmp_path):
    from mlops_backend import check

    snap = tmp_path / "hf" / "hub" / check._REPO_DIR / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "cosyvoice3.yaml").write_text("x")
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert check._model_dir() == snap
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "elsewhere"))
    assert check._model_dir() is None


def test_check_defaults_to_home_cache_without_hf_env(monkeypatch, tmp_path):
    from mlops_backend import check

    snap = tmp_path / ".cache" / "huggingface" / "hub" / check._REPO_DIR / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "cosyvoice3.yaml").write_text("x")
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(check.Path, "home", classmethod(lambda cls: tmp_path))
    assert check._model_dir() == snap
