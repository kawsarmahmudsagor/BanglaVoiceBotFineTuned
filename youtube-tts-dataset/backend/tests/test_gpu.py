import pytest

from ytds.gpu import GpuWorker, run_with_oom_retry


def test_use_swaps_and_empties_cache():
    calls = []
    w = GpuWorker(
        {"a": lambda: calls.append("load a") or "A", "b": lambda: calls.append("load b") or "B"},
        empty_cache=lambda: calls.append("empty"),
    )
    assert w.loaded is None
    assert w.use("a") == "A"
    assert w.loaded == "a"
    assert w.use("b") == "B"
    assert w.loaded == "b"
    assert calls == ["load a", "empty", "load b"]
    w.release()
    assert w.loaded is None
    assert calls[-1] == "empty"


def test_use_same_model_no_reload():
    n = []
    w = GpuWorker({"a": lambda: n.append(1) or object()}, empty_cache=lambda: None)
    assert w.use("a") is w.use("a")
    assert len(n) == 1


def test_unload_drops_only_the_named_model():
    calls = []
    w = GpuWorker({"a": lambda: "A", "b": lambda: "B"}, empty_cache=lambda: calls.append("empty"))
    w.unload("a")  # nothing loaded: no-op
    assert calls == [] and w.loaded is None
    w.use("a")
    w.unload("b")  # another model loaded: no-op
    assert w.loaded == "a" and calls == []
    w.unload("a")
    assert w.loaded is None and calls == ["empty"]


def test_oom_retry_halves_once():
    seen = []
    cleared = []

    def fn(x):
        seen.append(x)
        if len(seen) == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate")
        return "ok"

    assert run_with_oom_retry(fn, 8.0, empty_cache=lambda: cleared.append(1)) == "ok"
    assert seen == [8.0, 4.0]
    assert cleared == [1]


def test_second_oom_raises():
    def fn(x):
        raise RuntimeError("out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        run_with_oom_retry(fn, 8.0, empty_cache=lambda: None)


def test_non_oom_error_propagates_without_retry():
    seen = []

    def fn(x):
        seen.append(x)
        raise ValueError("boom")

    with pytest.raises(ValueError):
        run_with_oom_retry(fn, 8.0, empty_cache=lambda: None)
    assert seen == [8.0]
