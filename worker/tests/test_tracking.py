from types import SimpleNamespace

from mlops_shared.contract import Metric
from mlops_worker import __main__ as worker_main
from mlops_worker.executor import NullForwarder
from mlops_worker.tracking import MlflowForwarder, flatten, make_forwarder

TAGS = {"backend": "unsloth", "base_model": "m", "commercial_ok": "true", "git_sha": "abc", "task": "asr"}
PARAMS = {"method": {"r": 16}, "hparams": {"lr": 0.1}, "base_model": "m", "export": ["gguf"]}


class StubClient:
    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    def _rec(self, name, *a, **kw):
        if name in self.fail_on:
            raise RuntimeError(f"{name} down")
        self.calls.append((name, a, kw))

    def get_experiment_by_name(self, name):
        self._rec("get_experiment_by_name", name)
        return None

    def create_experiment(self, name):
        self._rec("create_experiment", name)
        return "exp-1"

    def create_run(self, experiment_id, run_name=None, tags=None):
        self._rec("create_run", experiment_id, run_name=run_name, tags=tags)
        return SimpleNamespace(info=SimpleNamespace(run_id="mlf-1"))

    def log_batch(self, run_id, metrics=(), params=(), tags=()):
        self._rec("log_batch", run_id, metrics=list(metrics), params=list(params), tags=list(tags))

    def log_metric(self, run_id, key, value, timestamp=None, step=None):
        self._rec("log_metric", run_id, key, value, timestamp=timestamp, step=step)

    def set_terminated(self, run_id, status):
        self._rec("set_terminated", run_id, status)


def names(c):
    return [n for n, _, _ in c.calls]


def test_flatten():
    assert flatten({"method": {"r": 16}, "export": ["gguf"], "a": "x"}) == {
        "method.r": "16", "export": "['gguf']", "a": "x",
    }
    assert flatten({"a": {"b": {"c": 1}}}, "p") == {"p.a.b.c": "1"}


def test_start_creates_experiment_run_params_tags():
    c = StubClient()
    f = MlflowForwarder("http://x", client=c)
    assert f.start("run-1", PARAMS, TAGS) == "mlf-1"
    assert ("create_experiment", ("bangla-voice-asr",), {}) in c.calls
    cr = next(kw for n, _, kw in c.calls if n == "create_run")
    assert cr["run_name"] == "run-1"
    assert cr["tags"]["backend"] == "unsloth" and cr["tags"]["git_sha"] == "abc"
    batch = next(kw for n, _, kw in c.calls if n == "log_batch")
    assert {p.key: p.value for p in batch["params"]}["method.r"] == "16"


def test_params_truncated_to_500():
    c = StubClient()
    MlflowForwarder("u", client=c).start("r", {"big": "x" * 900}, TAGS)
    batch = next(kw for n, _, kw in c.calls if n == "log_batch")
    assert len(batch["params"][0].value) == 500


def test_metrics_key_step_timestamp():
    c = StubClient()
    f = MlflowForwarder("u", client=c)
    f.start("r", PARAMS, TAGS)
    f.metrics([Metric(step=3, split="train", name="loss", value=0.5, ts=2.0)])
    f.metrics([
        Metric(step=4, split="eval", name="wer", value=0.2, ts=3.0),
        Metric(step=5, split="eval", name="cer", value=0.1, ts=4.0),
    ])
    single = next(kw for n, a, kw in c.calls if n == "log_metric")
    assert (single["step"], single["timestamp"]) == (3, 2000)
    batches = [kw for n, _, kw in c.calls if n == "log_batch" and kw["metrics"]]
    assert [(m.key, m.step, m.timestamp) for m in batches[0]["metrics"]] == [
        ("eval/wer", 4, 3000), ("eval/cer", 5, 4000),
    ]
    assert any(n == "log_metric" and a[1] == "train/loss" for n, a, _ in c.calls)


def test_finish_status_mapping():
    for state, status in [("done", "FINISHED"), ("failed", "FAILED"), ("cancelled", "KILLED")]:
        c = StubClient()
        f = MlflowForwarder("u", client=c)
        f.start("r", PARAMS, TAGS)
        f.finish(state)
        assert ("set_terminated", ("mlf-1", status), {}) in c.calls


def test_log_metric_exception_swallowed():
    c = StubClient(fail_on={"log_metric", "log_batch"})
    f = MlflowForwarder("u", client=c)
    f.start("r", PARAMS, TAGS)
    f.metrics([Metric(step=1, split="train", name="loss", value=1.0, ts=1.0)])
    f.finish("done")


def test_failed_start_makes_everything_noop():
    c = StubClient(fail_on={"create_experiment"})
    f = MlflowForwarder("u", client=c)
    assert f.start("r", PARAMS, TAGS) is None
    f.metrics([Metric(step=1, split="train", name="loss", value=1.0, ts=1.0)])
    f.finish("done")
    assert "log_metric" not in names(c) and "set_terminated" not in names(c)


def test_construction_does_not_touch_network():
    MlflowForwarder("http://127.0.0.1:1")  # no client created, no raise


def test_make_forwarder():
    assert isinstance(make_forwarder(SimpleNamespace(mlflow_tracking_uri=None)), NullForwarder)
    f = make_forwarder(SimpleNamespace(mlflow_tracking_uri="http://m"))
    assert isinstance(f, MlflowForwarder)


def test_main_defaults_to_make_forwarder():
    import inspect

    assert inspect.signature(worker_main.main).parameters["forwarder_factory"].default is make_forwarder
