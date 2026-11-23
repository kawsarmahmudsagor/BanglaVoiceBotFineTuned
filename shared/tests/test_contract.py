from mlops_shared.contract import (
    DatasetRef,
    EvalResults,
    ExportArtifact,
    ExportManifest,
    Metric,
    Phase,
    RunConfig,
    RunDir,
    Status,
    append_metric,
    read_config,
    read_eval_results,
    read_export_manifest,
    read_metrics,
    read_status,
    write_config,
    write_eval_results,
    write_export_manifest,
    write_status,
)


def test_status_roundtrip_is_atomic(tmp_path):
    rd = RunDir(tmp_path / "r1"); rd.ensure()
    write_status(rd, Status(phase=Phase.train, step=3, total_steps=10, updated_at=1.0))
    assert read_status(rd).step == 3
    assert not list(rd.path.glob("*.tmp"))


def test_read_status_missing_or_partial_returns_none(tmp_path):
    rd = RunDir(tmp_path / "r1"); rd.ensure()
    assert read_status(rd) is None
    rd.status_path.write_text('{"phase": "tr')
    assert read_status(rd) is None


def test_read_metrics_ignores_trailing_partial_line(tmp_path):
    rd = RunDir(tmp_path / "r1"); rd.ensure()
    append_metric(rd, Metric(step=1, split="train", name="loss", value=0.5, ts=1.0))
    with rd.metrics_path.open("a") as f:
        f.write('{"step": 2, "spl')
    ms, offset = read_metrics(rd)
    assert [m.step for m in ms] == [1]
    with rd.metrics_path.open("a") as f:
        f.write('it": "train", "name": "loss", "value": 0.4, "ts": 2.0}\n')
    ms2, _ = read_metrics(rd, offset)
    assert [m.step for m in ms2] == [2]


def _cfg() -> RunConfig:
    return RunConfig(
        run_id="r1", attempt=1, resume=False, task="llm-sft", backend="fake",
        base_model="m", method={"type": "none"}, hparams={"steps": 5}, export=["dummy"],
        seed=42,
        datasets={"train": DatasetRef(name="d", version=1, hash="h", root="/d",
                                      manifest="/d/m.jsonl", splits=["train", "val"])},
        prepared_root="/p",
    )


def test_config_roundtrip(tmp_path):
    rd = RunDir(tmp_path / "r1"); rd.ensure()
    cfg = _cfg()
    write_config(rd, cfg)
    assert read_config(rd) == cfg
    assert not list(rd.path.glob("*.tmp"))


def test_layout_paths(tmp_path):
    rd = RunDir(tmp_path / "r1"); rd.ensure()
    assert rd.config_path.name == "config.json"
    assert rd.status_path.name == "status.json"
    assert rd.metrics_path.name == "metrics.jsonl"
    for d in (rd.logs_dir, rd.checkpoints_dir, rd.eval_dir, rd.export_dir):
        assert d.is_dir()
    assert [d.name for d in (rd.logs_dir, rd.checkpoints_dir, rd.eval_dir, rd.export_dir)] == [
        "logs", "checkpoints", "eval", "export"]


def test_eval_and_export_roundtrip_and_missing(tmp_path):
    rd = RunDir(tmp_path / "r1"); rd.ensure()
    assert read_eval_results(rd) is None
    assert read_export_manifest(rd) is None
    write_eval_results(rd, EvalResults(metrics={"wer": 0.1}))
    assert read_eval_results(rd).metrics == {"wer": 0.1}
    m = ExportManifest(artifacts=[ExportArtifact(format="dummy", path="export/x", size_bytes=3)])
    write_export_manifest(rd, m)
    assert read_export_manifest(rd) == m
