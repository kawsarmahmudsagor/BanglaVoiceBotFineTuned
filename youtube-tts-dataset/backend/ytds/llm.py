"""Minimal Ollama client (local daemon or ollama.com cloud) with retry/backoff and a circuit breaker."""
from __future__ import annotations

import threading
import time
from typing import Callable

import httpx

PUNCT_SYSTEM = (
    "You add punctuation to Bangla transcripts. Return the same words in the same order. "
    "Only add or change punctuation marks (। ? , ! and quotes). Do not fix spelling, do not translate, "
    "do not add or remove words. Output only the text."
)
TRANSLATE_SYSTEM = (
    "Translate the English sentence into natural spoken Bangla, as a Bangladeshi speaker would say it. "
    "Keep English product and brand names in Latin script. Output only the Bangla text."
)

BACKOFF = (1, 2, 4)
BREAKER_THRESHOLD = 3      # consecutive LlmUnavailable calls that open the circuit
BREAKER_COOLDOWN_S = 60.0  # while open, chat() fails at once without a network call


class LlmUnavailable(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, host: str, model: str, api_key: str | None,
                 transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep, timeout: float = 120.0,
                 clock: Callable[[], float] = time.monotonic):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.model = model
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()  # chat() is called from the pool threads and the GPU thread
        self._failures = 0
        self._open_until = 0.0
        self._http = httpx.Client(base_url=host.rstrip("/"), headers=headers, transport=transport, timeout=timeout)

    def show(self) -> bool:
        try:
            return self._http.post("/api/show", json={"model": self.model}).status_code == 200
        except httpx.HTTPError:
            return False

    def chat(self, system: str, user: str) -> str:
        """After BREAKER_THRESHOLD consecutive failures the circuit opens: for BREAKER_COOLDOWN_S every call
        raises LlmUnavailable at once, so a whole video falls back to no_punct / pending_llm quickly. After the
        cooldown one call probes again; a failure reopens the circuit, a success closes it."""
        with self._lock:
            if self._clock() < self._open_until:
                raise LlmUnavailable("circuit open: LLM failed repeatedly, retrying later")
        try:
            out = self._chat(system, user)
        except LlmUnavailable:
            with self._lock:
                self._failures += 1
                if self._failures >= BREAKER_THRESHOLD:
                    self._open_until = self._clock() + BREAKER_COOLDOWN_S
            raise
        with self._lock:
            self._failures, self._open_until = 0, 0.0
        return out

    def _chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        last = "no response"
        for attempt in range(len(BACKOFF) + 1):
            if attempt:
                self._sleep(BACKOFF[attempt - 1])
            try:
                r = self._http.post("/api/chat", json=payload)
            except httpx.TransportError as exc:
                last = f"transport error: {exc!r}"
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}"
                continue
            if r.status_code != 200:
                raise LlmUnavailable(f"HTTP {r.status_code}")
            try:
                return r.json()["message"]["content"].strip()
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                raise LlmUnavailable(f"malformed response: {exc!r}") from exc
        raise LlmUnavailable(last)
