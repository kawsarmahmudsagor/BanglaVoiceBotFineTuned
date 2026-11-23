"""Fake backend: `python -m mlops_backend {prepare|train|eval|export|serve} --run-dir DIR`."""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

from mlops_shared.contract import (
    EvalResults, ExportArtifact, ExportManifest, Metric, Phase, RunConfig, RunDir, Status,
    append_metric, read_config, write_eval_results, write_export_manifest, write_status,
)


def _status(rd: RunDir, phase: Phase, **kw) -> None:
    write_status(rd, Status(phase=phase, updated_at=time.time(), **kw))


def _maybe_fail(cfg: RunConfig, phase: str) -> None:
    if cfg.hparams.get("fail_phase") == phase:
        print(f"fake backend: forced failure in {phase}", file=sys.stderr)
        sys.exit(1)


def _last_checkpoint(rd: RunDir) -> int:
    steps = [int(p.stem.split("-")[1]) for p in rd.checkpoints_dir.glob("step-*.txt")]
    return max(steps, default=0)


def _write_checkpoint(rd: RunDir, step: int) -> None:
    rd.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (rd.checkpoints_dir / f"step-{step}.txt").write_text(f"step {step}\n")


def prepare(rd: RunDir, cfg: RunConfig) -> int:
    _status(rd, Phase.prepare, message="preparing")
    _maybe_fail(cfg, "prepare")
    return 0


def train(rd: RunDir, cfg: RunConfig) -> int:
    steps = int(cfg.hparams.get("steps", 5))
    sleep_s = float(cfg.hparams.get("step_sleep_s", 0.0))
    _status(rd, Phase.train, total_steps=steps, message="training")
    _maybe_fail(cfg, "train")
    if cfg.hparams.get("oom"):
        print("RuntimeError: CUDA out of memory. Tried to allocate 20.00 GiB", file=sys.stderr)
        return 1
    rd.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    state = {"step": _last_checkpoint(rd) if cfg.resume else 0}

    if cfg.hparams.get("ignore_sigterm"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        def on_term(signum, frame):
            _write_checkpoint(rd, state["step"])
            sys.exit(143)
        signal.signal(signal.SIGTERM, on_term)

    for n in range(state["step"] + 1, steps + 1):
        if sleep_s:
            time.sleep(sleep_s)
        append_metric(rd, Metric(step=n, split="train", name="loss", value=1 / (n + 1), ts=time.time()))
        state["step"] = n
        _write_checkpoint(rd, n)
        _status(rd, Phase.train, step=n, total_steps=steps, message="training")
    return 0


def eval_(rd: RunDir, cfg: RunConfig) -> int:
    _status(rd, Phase.eval, message="evaluating")
    _maybe_fail(cfg, "eval")
    write_eval_results(rd, EvalResults(metrics={"fake_score": 0.5}))
    return 0


def export(rd: RunDir, cfg: RunConfig) -> int:
    _status(rd, Phase.export, message="exporting")
    _maybe_fail(cfg, "export")
    rd.export_dir.mkdir(parents=True, exist_ok=True)
    f = rd.export_dir / "model.dummy"
    f.write_text("dummy model\n")
    write_export_manifest(
        rd, ExportManifest(artifacts=[ExportArtifact(format="dummy", path="model.dummy", size_bytes=f.stat().st_size)])
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mlops_backend")
    ap.add_argument("phase", choices=["prepare", "train", "eval", "export", "serve"])
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)
    if args.phase == "serve":
        print("fake backend cannot serve")
        return 2
    rd = RunDir(Path(args.run_dir))
    cfg = read_config(rd)
    return {"prepare": prepare, "train": train, "eval": eval_, "export": export}[args.phase](rd, cfg)


if __name__ == "__main__":
    sys.exit(main())
