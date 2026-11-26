"""Minimal Ollama client (local daemon or ollama.com cloud) with retry/backoff."""
from __future__ import annotations

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


class LlmUnavailable(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, host: str, model: str, api_key: str | None,
                 transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep, timeout: float = 120.0):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.model = model
        self._sleep = sleep
        self._http = httpx.Client(base_url=host.rstrip("/"), headers=headers, transport=transport, timeout=timeout)

    def show(self) -> bool:
        try:
            return self._http.post("/api/show", json={"model": self.model}).status_code == 200
        except httpx.HTTPError:
            return False

    def chat(self, system: str, user: str) -> str:
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
            except httpx.ConnectError as exc:
                last = f"connect error: {exc}"
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}"
                continue
            if r.status_code != 200:
                raise LlmUnavailable(f"HTTP {r.status_code}")
            return r.json()["message"]["content"].strip()
        raise LlmUnavailable(last)
