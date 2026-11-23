import math

import pytest

from app.core.contract import DatasetRef, RunConfig, RunDir, read_eval_results, read_eval_samples
from app.core.backend_kit import Reporter
from app.core.evaluation import (
    baseline_metrics, merge_results, run_asr_eval, run_llm_eval, run_tts_eval,
)


@pytest.fixture
def rd(tmp_path):
    r = RunDir(tmp_path / "run")
    r.ensure()
    return r


@pytest.fixture
def rep(rd):
    return Reporter(rd)


@pytest.fixture
def cfg(tmp_path):
    return RunConfig(run_id="r", attempt=1, resume=False, task="asr", backend="b", base_model="m",
                     method={}, hparams={}, export=[], seed=1, datasets={}, prepared_root=str(tmp_path / "prep"))


@pytest.fixture
def ds():
    return DatasetRef(name="d", version=1, hash="h", root="/d", manifest="m.jsonl", splits=["train"])


def test_llm_eval_rule_metrics(rd, rep):
    rows = [{"sample_id": "a", "messages": [{"role": "user", "content": "দাম?"}, {"role": "assistant", "content": "৫০০ টাকা"}]}]
    m = run_llm_eval(rd, rep, rows, lambda msgs: "**দাম** ৫০০ টাকা 😊", loss=0.5)
    assert m["markdown_rate"] == 1.0 and m["emoji_rate"] == 1.0
    assert m["perplexity"] == pytest.approx(math.exp(0.5))
    assert m["eval_loss"] == 0.5
    s = read_eval_samples(rd, 0, 10)[0][0]
    assert s == {"sample_id": "a", "prompt": rows[0]["messages"][:1], "reference": "৫০০ টাকা", "reply": "**দাম** ৫০০ টাকা 😊"}


def test_llm_eval_caps_samples(rd, rep):
    rows = [{"sample_id": str(i), "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
            for i in range(250)]
    calls = []
    run_llm_eval(rd, rep, rows, lambda msgs: calls.append(1) or "ok", loss=None)
    assert len(calls) == 200


def test_asr_eval_normalizes_before_scoring(rd, rep):
    rows = [{"sample_id": "a", "audio_path": "x", "text": "আমি যাব।", "duration_s": 2.0}]
    m = run_asr_eval(rd, rep, rows, lambda paths: ["আমি যাব"])
    assert m["wer"] == 0.0 and m["cer"] == 0.0 and "rtf" in m


def test_asr_eval_slices_skip_missing(rd, rep):
    rows = [{"sample_id": "a", "text": "x", "duration_s": 1, "audio_path": "a", "region": "dhaka"},
            {"sample_id": "b", "text": "y", "duration_s": 1, "audio_path": "b", "region": None}]
    m = run_asr_eval(rd, rep, rows, lambda p: ["x", "z"])
    assert m["wer/region=dhaka"] == 0.0 and not any("None" in k for k in m)


def test_prefix_applied_and_no_samples_for_baseline(rd, rep):
    rows = [{"sample_id": "a", "audio_path": "x", "text": "x", "duration_s": 1.0}]
    m = run_asr_eval(rd, rep, rows, lambda p: ["x"], prefix="base/")
    assert m and all(k.startswith("base/") for k in m)
    assert not rd.eval_samples_path.exists()
    lrows = [{"sample_id": "a", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}]
    m = run_llm_eval(rd, rep, lrows, lambda msgs: "x", loss=1.0, prefix="base/")
    assert all(k.startswith("base/") for k in m)
    assert not rd.eval_samples_path.exists()


def test_tts_eval_writes_wavs(rd, rep):
    np = pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    rows = [{"sample_id": "a", "text": "আমি"}]
    m = run_tts_eval(rd, rep, rows, lambda text, row: (np.zeros(24000, dtype="float32"), 24000), loss=0.3)
    assert (rd.eval_audio_dir / "a.wav").exists()
    assert "rtf" in m and m["eval_loss"] == 0.3


def test_baseline_cached(cfg, ds):
    calls = []
    f = lambda: calls.append(1) or {"wer": 0.4}
    assert baseline_metrics(cfg, ds, {"k": 1}, f) == {"base/wer": 0.4}
    assert baseline_metrics(cfg, ds, {"k": 1}, f) == {"base/wer": 0.4}
    assert calls == [1]
    baseline_metrics(cfg, ds, {"k": 2}, f)
    assert calls == [1, 1]


def test_baseline_key_changes_with_split_ratios(cfg, ds):
    from app.core.splits import SplitConfig, SplitRatios

    calls = []
    f = lambda: calls.append(1) or {"wer": 0.4}
    a = ds.model_copy(update={"split": SplitConfig()})
    b = ds.model_copy(update={"split": SplitConfig(ratios=SplitRatios(train=0.8, val=0.1, test=0.1))})
    baseline_metrics(cfg, a, {"k": 1}, f)
    baseline_metrics(cfg, a, {"k": 1}, f)
    baseline_metrics(cfg, b, {"k": 1}, f)
    assert calls == [1, 1]


def test_merge_results(rd, rep):
    merge_results(rd, {"a": 1.0}, {"b": 2.0})
    res = read_eval_results(rd)
    assert res.metrics == {"a": 1.0, "b": 2.0} and res.samples_dir is None
    rows = [{"sample_id": "a", "audio_path": "x", "text": "x", "duration_s": 1.0}]
    run_asr_eval(rd, rep, rows, lambda p: ["x"])
    merge_results(rd, {"a": 1.0})
    assert read_eval_results(rd).samples_dir == "eval"


def test_rerun_does_not_duplicate_samples(rd, rep):
    lrows = [{"sample_id": "a", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}]
    for _ in range(2):
        run_llm_eval(rd, rep, lrows, lambda m: "x", loss=None)
    assert read_eval_samples(rd, 0, 10)[1] == 1
    arows = [{"sample_id": "a", "audio_path": "x", "text": "x", "duration_s": 1.0}]
    for _ in range(2):
        run_asr_eval(rd, rep, arows, lambda p: ["x"])
    assert read_eval_samples(rd, 0, 10)[1] == 1


def test_rerun_clears_stale_tts_wavs(rd, rep):
    np = pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    syn = lambda t, r: (np.zeros(2400, dtype="float32"), 24000)
    run_tts_eval(rd, rep, [{"sample_id": "old", "text": "x"}], syn, loss=None)
    run_tts_eval(rd, rep, [{"sample_id": "new", "text": "y"}], syn, loss=None)
    assert [p.name for p in rd.eval_audio_dir.glob("*.wav")] == ["new.wav"]
    assert read_eval_samples(rd, 0, 10)[1] == 1
