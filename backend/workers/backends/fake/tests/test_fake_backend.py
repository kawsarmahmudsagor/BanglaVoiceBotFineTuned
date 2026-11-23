import json
import signal
import subprocess
import sys
import time

import pytest
from app.core.contract import (
    RunConfig, RunDir, read_config, read_eval_results, read_export_manifest,
    read_metrics, read_status, write_config,
)
from app.core.recipes import Capabilities, RecipeSpec, validate_recipe


def make_cfg(**hparams) -> RunConfig:
    return RunConfig(
        run_id="r1", attempt=1, resume=False, task="llm-sft", backend="fake",
        base_model="none", method={"type": "none"}, hparams=hparams, export=["dummy"],
        seed=1, datasets={}, prepared_root="prepared",
    )


def make_dir(tmp_path, **hparams) -> RunDir:
    rd = RunDir(tmp_path / "run")
    rd.ensure()
    write_config(rd, make_cfg(**hparams))
    return rd


def cmd(phase, rd):
    return [sys.executable, "-m", "mlops_backend", phase, "--run-dir", str(rd.path)]


def run(phase, rd):
    return subprocess.run(cmd(phase, rd), capture_output=True, text=True, timeout=60)


def test_full_pipeline_writes_contract_files(tmp_path):
    rd = make_dir(tmp_path)
    for phase in ["prepare", "train", "eval", "export"]:
        assert run(phase, rd).returncode == 0
    ms, _ = read_metrics(rd)
    assert [m.step for m in ms if m.name == "loss"] == [1, 2, 3, 4, 5]
    assert read_eval_results(rd).metrics == {"fake_score": 0.5}
    art = read_export_manifest(rd).artifacts[0]
    assert art.format == "dummy" and (rd.export_dir / art.path).exists()
    assert read_status(rd).phase.value == "export"


def test_fail_phase_exits_nonzero(tmp_path):
    rd = make_dir(tmp_path, fail_phase="train")
    assert run("prepare", rd).returncode == 0
    assert run("train", rd).returncode == 1


def test_oom_prints_cuda_oom_to_stderr(tmp_path):
    rd = make_dir(tmp_path, oom=True)
    r = run("train", rd)
    assert r.returncode == 1 and "CUDA out of memory" in r.stderr


def test_serve_unsupported(tmp_path):
    rd = make_dir(tmp_path)
    r = run("serve", rd)
    assert r.returncode == 2 and "fake backend cannot serve" in r.stdout


def test_capabilities_accept_valid_and_reject_bad_steps():
    from pathlib import Path
    import mlops_backend
    raw = json.loads((Path(mlops_backend.__file__).parent.parent / "capabilities.json").read_text())
    caps = {"fake": Capabilities.model_validate(raw)}
    assert set(caps["fake"].tasks) == {"llm-sft", "asr", "tts"}

    def spec(**hp):
        return RecipeSpec(
            name="x", task="asr", backend="fake", base_model="m",
            base_model_meta={"license": "mit", "commercial_ok": True},
            datasets={"train": "d@v1"}, method={"type": "none"}, hparams=hp, export=["dummy"],
        )
    assert validate_recipe(spec(steps=3), caps) == []
    assert validate_recipe(spec(steps=0), caps) != []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_sigterm_saves_checkpoint_and_resume_continues(tmp_path):
    rd = make_dir(tmp_path, steps=50, step_sleep_s=0.05)
    p = subprocess.Popen(cmd("train", rd), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 30
        while not list(rd.checkpoints_dir.glob("step-*")) and time.time() < deadline:
            time.sleep(0.02)
        p.send_signal(signal.SIGTERM)
        assert p.wait(5) == 143
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
    last = max(int(f.stem.split("-")[1]) for f in rd.checkpoints_dir.glob("step-*"))
    before, off = read_metrics(rd)
    cfg = read_config(rd)
    cfg.resume = True
    cfg.hparams["steps"] = last + 2
    write_config(rd, cfg)
    assert run("train", rd).returncode == 0
    new, _ = read_metrics(rd, off)
    assert min(m.step for m in new) == last + 1
    assert max(m.step for m in new) == last + 2
