"""Shared runtime for training backends (`python -m mlops_backend <phase> --run-dir DIR`).

Light on purpose: audio.py and hf_access.py are submodules and are not imported here.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, NoReturn

from app.core.contract import (
    Metric, Phase, RunConfig, RunDir, Status, append_metric, read_config, write_status,
)
from app.core.prepare import PrepareError

Handler = Callable[[RunDir, RunConfig, "Reporter"], int]

__all__ = [
    "Handler", "main", "Reporter", "StopFlag", "NanGuard", "TrainingDiverged", "mark_complete",
    "latest_checkpoint", "max_skip_share", "eval_opts", "INIT_META", "write_init_meta", "read_init_meta",
    "resolve_checkpoint", "InitChoice", "init_source",
]

INIT_META = "init_meta.json"


class TrainingDiverged(RuntimeError):
    pass


class Reporter:
    def __init__(self, rd: RunDir, clock: Callable[[], float] = time.time):
        self.rd = rd
        self.clock = clock
        self._phase: Phase | None = None
        self._phase_start = 0.0
        self._last_write = 0.0

    def status(self, phase: Phase, step: int = 0, total_steps: int | None = None, message: str = "") -> None:
        now = self.clock()
        changed = phase != self._phase
        if changed:
            self._phase = phase
            self._phase_start = now
        final = total_steps is not None and step == total_steps
        if not (changed or final or now - self._last_write >= 1.0):
            return
        eta = None
        if total_steps is not None and step > 0:
            eta = (now - self._phase_start) / step * max(total_steps - step, 0)
        write_status(self.rd, Status(phase=phase, step=step, total_steps=total_steps, eta_s=eta,
                                     message=message, updated_at=now))
        self._last_write = now

    def metric(self, step: int, split: str, name: str, value: float) -> None:
        append_metric(self.rd, Metric(step=step, split=split, name=name, value=value, ts=self.clock()))


class StopFlag:
    def __init__(self) -> None:
        self.requested = False

    def install(self) -> "StopFlag":
        signal.signal(signal.SIGTERM, self._handle)
        return self

    def _handle(self, signum, frame) -> None:
        self.requested = True


class NanGuard:
    def __init__(self, limit: int = 3) -> None:
        self.limit = limit
        self._bad = 0

    def feed(self, loss: float) -> None:
        if math.isfinite(loss):
            self._bad = 0
            return
        self._bad += 1
        if self._bad >= self.limit:
            raise TrainingDiverged("loss is NaN; lower learning_rate")


def mark_complete(ckpt_dir: Path) -> None:
    (Path(ckpt_dir) / ".complete").write_text("")


def latest_checkpoint(rd: RunDir) -> Path | None:
    best: tuple[int, Path] | None = None
    if not rd.checkpoints_dir.is_dir():
        return None
    for d in rd.checkpoints_dir.iterdir():
        m = re.search(r"(\d+)$", d.name)
        if not (d.is_dir() and m and (d / ".complete").exists()):
            continue
        n = int(m.group(1))
        if best is None or n > best[0]:
            best = (n, d)
    return best[1] if best else None


def write_init_meta(ckpt_dir: Path, *, backend: str, base_model: str, method: str, format: str, step: int) -> None:
    meta = {"backend": backend, "base_model": base_model, "method": method, "format": format, "step": step}
    (Path(ckpt_dir) / INIT_META).write_text(json.dumps(meta), encoding="utf-8")


def read_init_meta(ckpt_dir: Path) -> dict | None:
    try:
        data = json.loads((Path(ckpt_dir) / INIT_META).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def resolve_checkpoint(rd: RunDir, which: str | int) -> Path | None:
    """The complete checkpoint dir named by `which` ("final", "latest" or a step), else None."""
    if which == "latest":
        return latest_checkpoint(rd)
    if which == "final":
        d = rd.checkpoints_dir / "final"
        return d if (d / ".complete").exists() else None
    if not rd.checkpoints_dir.is_dir():
        return None
    for d in rd.checkpoints_dir.iterdir():
        m = re.search(r"(\d+)$", d.name)
        if d.is_dir() and m and int(m.group(1)) == which and (d / ".complete").exists():
            return d
    return None


@dataclass
class InitChoice:
    kind: Literal["resume", "init"]
    path: Path
    meta: dict | None


def init_source(cfg: RunConfig, rd: RunDir) -> InitChoice | None:
    """Where a train phase starts from: its own checkpoint when resuming, else the warm-start init dir."""
    if cfg.resume:
        own = latest_checkpoint(rd)
        if own is not None:
            return InitChoice("resume", own, read_init_meta(own))
    if cfg.init_from is not None:
        return InitChoice("init", rd.init_dir, read_init_meta(rd.init_dir))
    return None


def max_skip_share(cfg: RunConfig) -> float:
    prep = cfg.hparams.get("prepare") or {}
    return float(prep.get("max_skip_share", 0.05))


def eval_opts(cfg: RunConfig) -> tuple[bool, int | None]:
    ev = cfg.hparams.get("eval") or {}
    ms = ev.get("max_samples")
    return bool(ev.get("baseline", True)), (int(ms) if ms is not None else None)


def _fail_status(rd: RunDir, phase: Phase, message: str) -> None:
    write_status(rd, Status(phase=phase, message=message, updated_at=time.time()))


def main(handlers: dict[Phase, Handler], argv: list[str] | None = None) -> NoReturn:
    parser = argparse.ArgumentParser(prog="mlops_backend")
    parser.add_argument("phase")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)  # exits 2 on bad usage
    try:
        phase = Phase(args.phase)
        handler = handlers[phase]
    except (ValueError, KeyError):
        print(f"unknown phase: {args.phase}", file=sys.stderr)
        sys.exit(2)
    rd = RunDir(Path(args.run_dir))
    code = 1
    try:
        cfg = read_config(rd)
        code = int(handler(rd, cfg, Reporter(rd)))
    except PrepareError as e:
        traceback.print_exc()
        _fail_status(rd, phase, str(e))
    except Exception as e:  # noqa: BLE001 - top-level boundary
        traceback.print_exc()
        if type(e).__name__ == "OutOfMemoryError":
            print("RuntimeError: CUDA out of memory", file=sys.stderr)
        lines = str(e).splitlines()
        _fail_status(rd, phase, f"{type(e).__name__}: {lines[0] if lines else ''}")
    sys.exit(code)
