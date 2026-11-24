import json

import pytest

from app.core.contract import InitRef, read_config

W2V = "facebook/wav2vec2-xls-r-300m"
MMS = "facebook/mms-1b-all"


def _warm(run_dir, base, fmt, vocab=None):
    rd = run_dir("asr", base)
    cfg = read_config(rd).model_copy(update={"init_from": InitRef(
        path=str(rd.init_dir), source_run="r0", checkpoint="final",
        method="adapter" if fmt == "hf-mms-adapter" else "full", format=fmt)})
    rd.init_dir.mkdir(parents=True, exist_ok=True)
    (rd.init_dir / "init_meta.json").write_text(json.dumps({"format": fmt}))
    if vocab is not None:
        (rd.init_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    return rd, cfg


def test_dispatch_ctc_families(tmp_path, monkeypatch):
    from app.core.contract import Phase, RunConfig, RunDir
    from mlops_backend import ctc, phases

    assert phases._FAMILY_MODULE["ctc"] == "ctc" and phases._FAMILY_MODULE["ctc-mms"] == "ctc"
    seen = []
    for phase in ("prepare", "train", "eval", "export"):
        monkeypatch.setattr(ctc, phase, lambda rd, cfg, rep, p=phase: seen.append(p) or 0)
    rd = RunDir(tmp_path / "run")
    rd.ensure()
    for base in (W2V, MMS):
        cfg = RunConfig(run_id="r1", attempt=1, resume=False, task="asr", backend="hf", base_model=base,
                        method={"type": "full"}, hparams={}, export=[], seed=1, datasets={}, prepared_root="p")
        for p in Phase:
            if p.value in ("prepare", "train", "eval", "export"):
                assert phases._dispatch(p)(rd, cfg, None) == 0
    assert seen.count("train") == 2


def test_handler_names():
    from mlops_backend import ctc as c

    assert c.prepare is c.prepare_ctc and c.train is c.train_ctc
    assert c.eval is c.eval_ctc and c.export is c.export_ctc


def test_ctc_vocab_from_init_when_warm_start(run_dir):
    from mlops_backend.ctc import resolve_vocab

    init_vocab = {"আ": 0, "|": 1, "[UNK]": 2, "[PAD]": 3}
    rd, cfg = _warm(run_dir, W2V, "hf-full", init_vocab)
    vocab, mode = resolve_vocab(rd, cfg, lambda: pytest.fail("must not scan texts"), lambda: pytest.fail("no mms"))
    assert vocab == init_vocab and mode == "fixed"


def test_vocab_built_from_texts_when_cold(run_dir):
    from mlops_backend.ctc import resolve_vocab

    cfg = read_config(run_dir("asr", W2V))
    vocab, mode = resolve_vocab(run_dir("asr", W2V), cfg, lambda: ["আ আ"], lambda: pytest.fail("no mms"))
    assert mode == "new" and "আ" in vocab and "|" in vocab


def test_mms_cold_uses_base_vocab_fixed(run_dir):
    from mlops_backend.ctc import resolve_vocab

    rd = run_dir("asr", MMS)
    vocab, mode = resolve_vocab(rd, read_config(rd), lambda: pytest.fail("no texts"), lambda: {"a": 0})
    assert vocab == {"a": 0} and mode == "fixed"


def test_warm_start_without_vocab_fails(run_dir):
    from app.core.prepare import PrepareError
    from mlops_backend.ctc import resolve_vocab

    rd, cfg = _warm(run_dir, W2V, "hf-full")
    with pytest.raises(PrepareError, match="no vocab.json"):
        resolve_vocab(rd, cfg, lambda: [], lambda: {})


def test_init_plan():
    from mlops_backend.ctc import plan_ctc_init

    assert plan_ctc_init(None, False) == "base"
    assert plan_ctc_init({"format": "hf-full"}, False) == "full_weights"
    assert plan_ctc_init({"format": "hf-mms-adapter"}, True) == "adapter"
    with pytest.raises(ValueError, match="hf-mms-adapter.*only hf-full"):
        plan_ctc_init({"format": "hf-mms-adapter"}, False)
    with pytest.raises(ValueError, match="only hf-mms-adapter"):
        plan_ctc_init({"format": "hf-full"}, True)


@pytest.mark.parametrize("base,method,fmt", [(W2V, "full", "hf-full"), (MMS, "adapter", "hf-mms-adapter")])
def test_ckpt_meta(run_dir, tmp_path, base, method, fmt):
    from mlops_backend.ctc import write_ckpt_meta

    cfg = read_config(run_dir("asr", base))
    d = tmp_path / "ck"
    d.mkdir()
    write_ckpt_meta(d, cfg, 7)
    meta = json.loads((d / "init_meta.json").read_text())
    assert meta["format"] == fmt and meta["method"] == method and meta["step"] == 7


@pytest.mark.parametrize("base,formats", [(W2V, {"safetensors"}), (MMS, {"safetensors", "adapter-safetensors"})])
def test_exporters_per_family(run_dir, base, formats):
    from mlops_backend.ctc import ctc_exporters

    rd = run_dir("asr", base)
    assert set(ctc_exporters(rd, read_config(rd))) == formats


def test_exporter_requires_complete_final(run_dir):
    from mlops_backend.ctc import ctc_exporters

    rd = run_dir("asr", W2V)
    (rd.checkpoints_dir / "final").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no completed final checkpoint"):
        ctc_exporters(rd, read_config(rd))["safetensors"](rd.export_dir / "safetensors")


def test_ct2_is_unknown_format_for_ctc(run_dir):
    from app.core.backend_kit import export_all
    from mlops_backend.ctc import ctc_exporters

    class Rep:
        def status(self, *a, **k): ...

    rd = run_dir("asr", W2V, export=["ct2"])
    assert export_all(rd, read_config(rd), Rep(), ctc_exporters(rd, read_config(rd))) == 1


def test_eval_requires_complete_final(run_dir):
    from mlops_backend.ctc import eval_ctc

    rd = run_dir("asr", W2V)
    (rd.checkpoints_dir / "final").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no completed final checkpoint"):
        eval_ctc(rd, read_config(rd), None)


def test_trainer_args(run_dir):
    from mlops_backend.ctc import ctc_trainer_args

    rd = run_dir("asr", W2V, {"max_steps": 20, "batch_size": 1})
    rd.prepared_refs_path.write_text('[{"role": "train", "path": "/p", "counts": {"train": 8, "val": 2}}]')
    a = ctc_trainer_args(read_config(rd), rd)
    assert a["max_steps"] == 20 and a["label_names"] == ["labels"] and a["remove_unused_columns"] is False
    assert a["gradient_checkpointing"] is True and a["learning_rate"] == 1e-4


def test_load_train_val_checks_refs(run_dir, tmp_path):
    from mlops_backend.ctc import load_train_val

    rd = run_dir("asr", W2V)
    p = tmp_path / "prep"
    p.mkdir()
    (p / "train.jsonl").write_text(json.dumps({"audio_path": str(tmp_path / "gone.wav"), "duration_s": 1, "text": "x"}) + "\n")
    (p / "val.jsonl").write_text("")
    rd.prepared_refs_path.write_text(json.dumps([{"role": "train", "path": str(p), "counts": {}}]))
    with pytest.raises(RuntimeError, match="points at missing audio"):
        load_train_val(rd)


@pytest.mark.heavy
def test_wav2vec2_20_steps(run_dir, monkeypatch, tmp_path):
    """GPU VM only: 20 steps, eval with baseline, safetensors export."""
    from app.core.backend_kit import Reporter
    from app.core.contract import read_eval_results
    from mlops_backend import ctc as c

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    rd = run_dir("asr", W2V, {"max_steps": 20, "batch_size": 1, "grad_accum": 1, "learning_rate": 1e-4},
                 export=["safetensors"])
    cfg = read_config(rd)
    rep = Reporter(rd)
    assert c.prepare(rd, cfg, rep) == 0
    assert c.train(rd, cfg, rep) == 0
    final = rd.checkpoints_dir / "final"
    assert (final / ".complete").exists()
    assert json.loads((final / "init_meta.json").read_text())["format"] == "hf-full"
    assert (final / "vocab.json").exists()
    assert c.eval(rd, cfg, rep) == 0
    res = read_eval_results(rd).metrics
    assert "base/wer" in res and "wer" in res
    assert c.export(rd, cfg, rep) == 0
    assert list((rd.export_dir / "safetensors").glob("*.safetensors"))
    assert rd.export_manifest_path.exists()


@pytest.mark.heavy
def test_mms_adapter_only_trains_adapter(run_dir, monkeypatch, tmp_path):
    from app.core.backend_kit import Reporter
    from mlops_backend import ctc as c

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    rd = run_dir("asr", MMS, {"max_steps": 5, "batch_size": 1, "grad_accum": 1},
                 export=["safetensors", "adapter-safetensors"], method={"type": "adapter"})
    cfg = read_config(rd)
    rep = Reporter(rd)
    assert c.prepare(rd, cfg, rep) == 0
    model = c.new_model(cfg, rd, c.read_vocab(rd), train=True)
    assert c.trainable_param_count(model) == c.adapter_param_count(model) > 0
    n_tensors = sum(1 for p in model.parameters() if p.requires_grad)
    assert n_tensors == len(model._get_adapters())
    del model
    assert c.train(rd, cfg, rep) == 0
    final = rd.checkpoints_dir / "final"
    assert (final / "adapter.ben.safetensors").exists()
    assert json.loads((final / "init_meta.json").read_text())["format"] == "hf-mms-adapter"
    assert c.export(rd, cfg, rep) == 0
    assert (rd.export_dir / "adapter-safetensors" / "adapter.ben.safetensors").exists()
