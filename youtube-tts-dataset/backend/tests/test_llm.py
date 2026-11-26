import httpx
import pytest

from ytds.llm import LlmUnavailable, OllamaClient


def _client(handler, key=None, sleeps=None):
    return OllamaClient("http://x", "m", key, transport=httpx.MockTransport(handler),
                        sleep=(sleeps.append if sleeps is not None else (lambda s: None)))


def _ok(request):
    return httpx.Response(200, json={"message": {"content": " hi "}})


def test_bearer_only_with_key():
    seen = []

    def handler(request):
        seen.append(request.headers.get("authorization"))
        return _ok(request)

    assert _client(handler, key="k").chat("s", "u") == "hi"
    assert _client(handler).chat("s", "u") == "hi"
    assert seen == ["Bearer k", None]


def test_chat_payload():
    import json

    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/api/chat"
        assert body["model"] == "m" and body["stream"] is False
        assert body["options"]["temperature"] == 0
        assert [m["role"] for m in body["messages"]] == ["system", "user"]
        return _ok(request)

    _client(handler).chat("s", "u")


def test_retries_then_unavailable():
    sleeps: list[float] = []
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503)

    with pytest.raises(LlmUnavailable):
        _client(handler, sleeps=sleeps).chat("s", "u")
    assert sleeps == [1, 2, 4]
    assert len(calls) == 4


def test_retries_connect_error_then_succeeds():
    n = {"i": 0}

    def handler(request):
        n["i"] += 1
        if n["i"] < 3:
            raise httpx.ConnectError("down")
        return _ok(request)

    assert _client(handler).chat("s", "u") == "hi"


def test_client_error_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400)

    with pytest.raises(LlmUnavailable):
        _client(handler).chat("s", "u")
    assert len(calls) == 1


def test_show_false_on_404():
    assert _client(lambda r: httpx.Response(404)).show() is False
    assert _client(lambda r: httpx.Response(200, json={})).show() is True


@pytest.mark.network
def test_live_punctuate():
    from ytds.config import load_settings
    from ytds.pipeline.punct import punctuate

    s = load_settings()
    llm = OllamaClient(s.ollama_host, s.llm_model, s.ollama_api_key or None)
    out, no_punct = punctuate("আমি ভাত খাই তুমি কি খাও", llm)
    print(ascii(out), no_punct)
    assert out


def test_timeout_retries_then_unavailable():
    sleeps: list[float] = []

    def handler(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(LlmUnavailable):
        _client(handler, sleeps=sleeps).chat("s", "u")
    assert sleeps == [1, 2, 4]


@pytest.mark.parametrize("resp", [
    httpx.Response(200, content=b"not json"),
    httpx.Response(200, json={"oops": 1}),
    httpx.Response(200, json=[1]),
])
def test_malformed_200_is_unavailable(resp):
    with pytest.raises(LlmUnavailable):
        _client(lambda r: resp).chat("s", "u")


def test_429_retried():
    n = {"i": 0}

    def handler(request):
        n["i"] += 1
        return httpx.Response(429) if n["i"] == 1 else _ok(request)

    assert _client(handler).chat("s", "u") == "hi"
    assert n["i"] == 2


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_circuit_opens_after_consecutive_failures_and_recovers_after_cooldown():
    from ytds.llm import BREAKER_COOLDOWN_S, BREAKER_THRESHOLD

    clock, calls, state = Clock(), [], {"up": False}

    def handler(request):
        calls.append(1)
        return _ok(request) if state["up"] else httpx.Response(402)

    c = OllamaClient("http://x", "m", None, transport=httpx.MockTransport(handler), sleep=lambda s: None,
                     clock=clock)
    for _ in range(BREAKER_THRESHOLD):
        with pytest.raises(LlmUnavailable):
            c.chat("s", "u")
    assert len(calls) == BREAKER_THRESHOLD

    with pytest.raises(LlmUnavailable, match="circuit open"):  # fails fast, no network call
        c.chat("s", "u")
    assert len(calls) == BREAKER_THRESHOLD

    clock.t += BREAKER_COOLDOWN_S + 1  # half-open: one probe goes out and fails, so it reopens
    with pytest.raises(LlmUnavailable):
        c.chat("s", "u")
    assert len(calls) == BREAKER_THRESHOLD + 1
    with pytest.raises(LlmUnavailable, match="circuit open"):
        c.chat("s", "u")

    clock.t += BREAKER_COOLDOWN_S + 1
    state["up"] = True
    assert c.chat("s", "u") == "hi"
    assert c.chat("s", "u") == "hi"


def test_success_resets_failure_count():
    from ytds.llm import BREAKER_THRESHOLD

    seq = iter([402] * (BREAKER_THRESHOLD - 1) + [200] + [402] * (BREAKER_THRESHOLD - 1) + [200])

    def handler(request):
        code = next(seq)
        return _ok(request) if code == 200 else httpx.Response(code)

    c = _client(handler)
    for _ in range(2):
        for _ in range(BREAKER_THRESHOLD - 1):
            with pytest.raises(LlmUnavailable):
                c.chat("s", "u")
        assert c.chat("s", "u") == "hi"  # never reached the threshold in a row
