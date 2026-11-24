import signal
import sys

import pytest

from app.core.backend_kit import (
    InitChoice, NanGuard, Reporter, StopFlag, TrainingDiverged, eval_opts, init_source, latest_checkpoint,
    main, mark_complete, max_skip_share, read_init_meta, resolve_checkpoint, write_init_meta,
)
from app.core.contract import (
    InitRef, Phase, RunConfig, RunDir, append_eval_sample, read_eval_samples, read_metrics,
    read_status, write_config,
)
from app.core.prepare import PrepareError


def _cfg(**hp):
    return RunConfig(run_id="r", attempt=1, resume=False, task="llm", backend="b", base_model="m",
                     method={}, hparams=hp, export=[], seed=1, datasets={}, prepared_root="/x")


@pytest.fixture
def run_dir_with_config(tmp_path):
    rd = RunDir(tmp_path)
    rd.ensure()
    write_config(rd, _cfg())
    return tmp_path


def test_main_dispatches_and_exits_with_handler_code(run_dir_with_config):
    with pytest.raises(SystemExit) as e:
        main({Phase.train: lambda rd, cfg, rep: 7}, ["train", "--run-dir", str(run_dir_with_config)])
    assert e.value.code == 7


def test_main_unknown_phase_exits_2(run_dir_with_config):
    with pytest.raises(SystemExit) as e:
        main({Phase.train: lambda rd, cfg, rep: 0}, ["export", "--run-dir", str(run_dir_with_config)])
    assert e.value.code == 2


def test_main_exception_writes_status_message(run_dir_with_config):
    def bad(rd, cfg, rep):
        raise ValueError("first line\nsecond")
    with pytest.raises(SystemExit) as e:
        main({Phase.train: bad}, ["train", "--run-dir", str(run_dir_with_config)])
    assert e.value.code == 1
    assert read_status(RunDir(run_dir_with_config)).message == "ValueError: first line"


def test_main_prepare_error_message(run_dir_with_config):
    def bad(rd, cfg, rep):
        raise PrepareError("no access to x/y: accept its terms on Hugging Face and set HF_TOKEN")
    with pytest.raises(SystemExit) as e:
        main({Phase.prepare: bad}, ["prepare", "--run-dir", str(run_dir_with_config)])
    assert e.value.code == 1
    assert read_status(RunDir(run_dir_with_config)).message.startswith("no access to x/y")


def test_main_nan_message_is_exact(run_dir_with_config):
    def bad(rd, cfg, rep):
        g = NanGuard()
        for _ in range(3):
            g.feed(float("nan"))
    with pytest.raises(SystemExit) as e:
        main({Phase.train: bad}, ["train", "--run-dir", str(run_dir_with_config)])
    assert e.value.code == 1
    assert read_status(RunDir(run_dir_with_config)).message == "loss is NaN; lower learning_rate"


def test_main_oom_prints_marker(run_dir_with_config, capsys):
    class OutOfMemoryError(RuntimeError): ...

    def oom(rd, cfg, rep):
        raise OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB")
    with pytest.raises(SystemExit) as e:
        main({Phase.train: oom}, ["train", "--run-dir", str(run_dir_with_config)])
    assert e.value.code == 1
    assert "RuntimeError: CUDA out of memory" in capsys.readouterr().err


def test_reporter_rate_limits(tmp_path):
    t = [0.0]
    rep = Reporter(RunDir(tmp_path), clock=lambda: t[0])
    rep.status(Phase.train, 1, 10)
    t[0] = 0.5
    rep.status(Phase.train, 2, 10)
    assert read_status(RunDir(tmp_path)).step == 1
    t[0] = 1.2
    rep.status(Phase.train, 3, 10)
    st = read_status(RunDir(tmp_path))
    assert st.step == 3 and st.eta_s == pytest.approx(2.8, rel=0.05)


def test_reporter_writes_final_step_immediately(tmp_path):
    t = [0.0]
    rep = Reporter(RunDir(tmp_path), clock=lambda: t[0])
    rep.status(Phase.train, 1, 10)
    t[0] = 0.1
    rep.status(Phase.train, 10, 10)
    assert read_status(RunDir(tmp_path)).step == 10


def test_reporter_phase_change_writes_immediately(tmp_path):
    t = [0.0]
    rep = Reporter(RunDir(tmp_path), clock=lambda: t[0])
    rep.status(Phase.train, 1, 10)
    t[0] = 0.1
    rep.status(Phase.eval, 0, None, "eval")
    assert read_status(RunDir(tmp_path)).phase == Phase.eval


def test_reporter_metric(tmp_path):
    rep = Reporter(RunDir(tmp_path))
    rep.metric(3, "train", "loss", 0.5)
    ms, _ = read_metrics(RunDir(tmp_path))
    assert ms[0].name == "loss" and ms[0].step == 3 and ms[0].value == 0.5


def test_nan_guard_trips_on_third():
    g = NanGuard()
    g.feed(float("nan"))
    g.feed(float("inf"))
    with pytest.raises(TrainingDiverged, match="loss is NaN; lower learning_rate"):
        g.feed(float("nan"))


def test_nan_guard_resets_on_finite():
    g = NanGuard()
    g.feed(float("nan"))
    g.feed(float("nan"))
    g.feed(1.0)
    g.feed(float("nan"))
    g.feed(float("nan"))


def test_latest_checkpoint_ignores_incomplete(tmp_path):
    rd = RunDir(tmp_path)
    rd.ensure()
    for n in (100, 200):
        (rd.checkpoints_dir / f"checkpoint-{n}").mkdir()
    assert latest_checkpoint(rd) is None
    mark_complete(rd.checkpoints_dir / "checkpoint-100")
    assert latest_checkpoint(rd).name == "checkpoint-100"
    mark_complete(rd.checkpoints_dir / "checkpoint-200")
    assert latest_checkpoint(rd).name == "checkpoint-200"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_stop_flag_on_sigterm():
    import os
    flag = StopFlag().install()
    assert flag.requested is False
    os.kill(os.getpid(), signal.SIGTERM)
    assert flag.requested is True


def test_max_skip_share_override():
    assert max_skip_share(_cfg()) == 0.05
    assert max_skip_share(_cfg(prepare={"max_skip_share": 0.2})) == 0.2


def test_eval_opts():
    assert eval_opts(_cfg()) == (True, None)
    assert eval_opts(_cfg(eval={"baseline": False, "max_samples": 5})) == (False, 5)


def test_eval_samples_roundtrip(tmp_path):
    rd = RunDir(tmp_path)
    for i in range(3):
        append_eval_sample(rd, {"i": i, "text": "bangla"})
    rows, total = read_eval_samples(rd, 1, 1)
    assert rows == [{"i": 1, "text": "bangla"}] and total == 3
    assert read_eval_samples(RunDir(tmp_path / "none"), 0, 5) == ([], 0)


def _ckpt(rd, name, complete=True):
    d = rd.checkpoints_dir / name
    d.mkdir(parents=True)
    if complete:
        mark_complete(d)
    return d


def test_resolve_checkpoint_final_latest_step(tmp_path):
    rd = RunDir(tmp_path)
    rd.ensure()
    assert resolve_checkpoint(rd, "final") is None and resolve_checkpoint(rd, "latest") is None
    _ckpt(rd, "checkpoint-100")
    c300 = _ckpt(rd, "checkpoint-300")
    _ckpt(rd, "checkpoint-500", complete=False)
    assert resolve_checkpoint(rd, "latest") == c300
    assert resolve_checkpoint(rd, 300) == c300
    assert resolve_checkpoint(rd, 500) is None and resolve_checkpoint(rd, 7) is None
    _ckpt(rd, "final", complete=False)
    assert resolve_checkpoint(rd, "final") is None
    mark_complete(rd.checkpoints_dir / "final")
    assert resolve_checkpoint(rd, "final") == rd.checkpoints_dir / "final"


def test_init_meta_roundtrip(tmp_path):
    assert read_init_meta(tmp_path) is None
    write_init_meta(tmp_path, backend="hf", base_model="m", method="lora", format="hf-peft", step=12)
    assert read_init_meta(tmp_path) == {"backend": "hf", "base_model": "m", "method": "lora",
                                        "format": "hf-peft", "step": 12}


def _init_cfg(rd, resume):
    cfg = _cfg()
    cfg.resume = resume
    cfg.init_from = InitRef(path=str(rd.init_dir), source_run="r-0", checkpoint="final", method="lora", format="x")
    return cfg


def test_init_source_prefers_own_checkpoint_on_resume(tmp_path):
    rd = RunDir(tmp_path)
    rd.ensure()
    own = _ckpt(rd, "checkpoint-5")
    assert init_source(_init_cfg(rd, True), rd) == InitChoice("resume", own, read_init_meta(own))
    assert init_source(_init_cfg(rd, False), rd).kind == "init"  # not resuming: own checkpoint ignored


def test_init_source_uses_init_dir_when_no_own_checkpoint(tmp_path):
    rd = RunDir(tmp_path)
    rd.ensure()
    rd.init_dir.mkdir()
    write_init_meta(rd.init_dir, backend="b", base_model="m", method="lora", format="x", step=1)
    ch = init_source(_init_cfg(rd, True), rd)
    assert ch.kind == "init" and ch.path == rd.init_dir and ch.meta["step"] == 1
    assert init_source(_cfg(), rd) is None


def _export_cfg(formats):
    cfg = _cfg()
    cfg.export = formats
    return cfg


def _art(fmt):
    from app.core.contract import ExportArtifact
    return lambda out: [ExportArtifact(format=fmt, path=f"{fmt}/m.bin", size_bytes=1)]


def _boom(out):
    raise RuntimeError("boom")


def test_export_all_partial_failure(tmp_path, capsys):
    from app.core.backend_kit import export_all
    from app.core.contract import read_export_manifest
    rd = RunDir(tmp_path)
    rd.ensure()
    rep = Reporter(rd)
    code = export_all(rd, _export_cfg(["merged-safetensors", "gguf"]), rep,
                      {"merged-safetensors": _art("merged-safetensors"), "gguf": _boom})
    assert code == 0
    assert [a.format for a in read_export_manifest(rd).artifacts] == ["merged-safetensors"]
    assert read_status(rd).message == "exported: merged-safetensors; failed: gguf"
    assert "boom" in capsys.readouterr().err


def test_export_all_all_succeed_message(tmp_path):
    from app.core.backend_kit import export_all
    rd = RunDir(tmp_path)
    rd.ensure()
    assert export_all(rd, _export_cfg(["a", "b"]), Reporter(rd), {"a": _art("a"), "b": _art("b")}) == 0
    assert read_status(rd).message == "exported: a, b"


def test_export_all_all_fail_exits_1(tmp_path):
    from app.core.backend_kit import export_all
    from app.core.contract import read_export_manifest
    rd = RunDir(tmp_path)
    rd.ensure()
    assert export_all(rd, _export_cfg(["a", "b"]), Reporter(rd), {"a": _boom, "b": _boom}) == 1
    assert read_export_manifest(rd) is None


def test_export_all_unknown_format_is_a_failure(tmp_path):
    from app.core.backend_kit import export_all
    rd = RunDir(tmp_path)
    rd.ensure()
    assert export_all(rd, _export_cfg(["nope"]), Reporter(rd), {"a": _art("a")}) == 1
    code = export_all(rd, _export_cfg(["a", "nope"]), Reporter(rd), {"a": _art("a")})
    assert code == 0 and read_status(rd).message == "exported: a; failed: nope"


def test_export_all_failed_reexport_removes_stale_manifest(tmp_path):
    from app.core.backend_kit import export_all
    from app.core.contract import read_export_manifest
    rd = RunDir(tmp_path)
    rd.ensure()
    assert export_all(rd, _export_cfg(["a"]), Reporter(rd), {"a": _art("a")}) == 0
    assert read_export_manifest(rd) is not None
    assert export_all(rd, _export_cfg(["a"]), Reporter(rd), {"a": _boom}) == 1
    assert read_export_manifest(rd) is None


def test_export_all_failed_exporter_leaves_no_dir_and_fresh_dir(tmp_path):
    from app.core.backend_kit import export_all
    rd = RunDir(tmp_path)
    rd.ensure()
    (rd.export_dir / "a").mkdir(parents=True)
    (rd.export_dir / "a" / "stale.bin").write_text("x")
    seen = {}

    def ok(out):
        seen["stale"] = (out / "stale.bin").exists()
        return _art("a")(out)

    def partial(out):
        (out / "half.bin").write_text("x")
        raise RuntimeError("boom")

    assert export_all(rd, _export_cfg(["a", "b"]), Reporter(rd), {"a": ok, "b": partial}) == 0
    assert seen["stale"] is False
    assert not (rd.export_dir / "b").exists()


def _plain_ckpt(root, name, complete=True):
    d = root / name
    d.mkdir(parents=True)
    if complete:
        mark_complete(d)
    return d


def test_prune_checkpoints_keeps_latest_k_complete(tmp_path):
    from app.core.backend_kit import keep_checkpoints, prune_checkpoints
    for n in (1, 2, 3, 4):
        _plain_ckpt(tmp_path, f"step-{n}")
    _plain_ckpt(tmp_path, "step-0", complete=False)  # stale incomplete, older than newest complete
    _plain_ckpt(tmp_path, "step-9", complete=False)  # in progress, newer: untouched
    _plain_ckpt(tmp_path, "final")
    _plain_ckpt(tmp_path, "init")
    _plain_ckpt(tmp_path, "other-1")
    prune_checkpoints(tmp_path, r"step-(\d+)", 2)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["final", "init", "other-1", "step-3", "step-4", "step-9"]
    prune_checkpoints(tmp_path, r"step-(\d+)", 0)  # the newest complete is never removed
    assert (tmp_path / "step-4").is_dir() and not (tmp_path / "step-3").exists()
    prune_checkpoints(tmp_path / "missing", r"step-(\d+)", 2)  # no dir: no-op
    assert keep_checkpoints({}) == 2 and keep_checkpoints({"keep_checkpoints": 5}) == 5
    assert keep_checkpoints({"keep_checkpoints": 0}) == 1
