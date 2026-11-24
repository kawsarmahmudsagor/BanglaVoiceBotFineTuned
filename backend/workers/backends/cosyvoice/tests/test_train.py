"""CosyVoice train / eval / export (light: no torch / cosyvoice / network, except the heavy-marked test)."""
import json

import pytest

from app.core.backend_kit import read_init_meta
from app.core.contract import read_config

from mlops_backend.export import link_or_copy
from mlops_backend.eval import split_sentences
from mlops_backend.train import (
    CKPT_FORMAT, assemble_final, checkpoint_dirs, clean_state, component_sources, epoch_dir, keep_count,
    latest_complete, plan_components, prune_checkpoints, publish_checkpoint, resolve_components,
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
    assert resolve_components({}) == ["llm", "flow"]
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
    assert meta["format"] == CKPT_FORMAT == "cosyvoice-pt" and meta["backend"] == "cosyvoice" and meta["step"] == 40


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


# ---- text --------------------------------------------------------------------------------------------------------

def test_split_sentences_on_danda():
    assert split_sentences("আমি যাব। তুমি এসো। শেষ") == ["আমি যাব।", "তুমি এসো।", "শেষ"]
    assert split_sentences("একটি বাক্য") == ["একটি বাক্য"]
    assert split_sentences("") == []


# ---- heavy (GPU VM only) -----------------------------------------------------------------------------------------

@pytest.mark.heavy
def test_cosyvoice_20_steps_llm_and_synthesize(run_dir, tiny_tts_ds, monkeypatch, tmp_path):
    from app.core.backend_kit import Reporter
    from app.core.contract import read_eval_results

    from mlops_backend import cosyvoice as p

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    _, rows = tiny_tts_ds(n=4, seconds=1.5)
    rd = run_dir("tts", MODEL, {"components": ["llm"], "max_steps": 20, "grad_accum": 1, "learning_rate": 1e-5,
                                "warmup_steps": 0, "save_steps": 10, "keep_checkpoints": 2,
                                "eval": {"max_samples": 2}}, export=["cosyvoice-dir"])
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
