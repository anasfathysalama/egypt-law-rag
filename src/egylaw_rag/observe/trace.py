"""One trace per /ask. Langfuse receives it when the host and keys are set."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import httpx


class AskTracer(Protocol):
    def span(self, name: str) -> None: ...

    def score(self, name: str, value: float) -> None: ...


class MemoryTracer:
    """Records spans in memory. Used in tests and when Langfuse is not configured."""

    def __init__(self) -> None:
        self.spans: list[str] = []
        self.scores: dict[str, float] = {}

    def span(self, name: str) -> None:
        self.spans.append(name)

    def score(self, name: str, value: float) -> None:
        self.scores[name] = value


class LangfuseTracer:
    """Sends each span and the faithfulness score to a self-hosted Langfuse."""

    def __init__(
        self,
        host: str,
        public_key: str,
        secret_key: str,
        post: Callable[[str, dict[str, object]], None] | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.public_key = public_key
        self.secret_key = secret_key
        self.spans: list[str] = []
        self.scores: dict[str, float] = {}
        self._post = post

    def span(self, name: str) -> None:
        self.spans.append(name)
        self._send({"type": "span", "name": name})

    def score(self, name: str, value: float) -> None:
        self.scores[name] = value
        self._send({"type": "score", "name": name, "value": value})

    def _send(self, payload: dict[str, object]) -> None:
        if self._post is not None:
            self._post(self.host, payload)
            return
        try:
            httpx.post(
                f"{self.host}/api/public/ingestion",
                json=payload,
                auth=(self.public_key, self.secret_key),
                timeout=2.0,
            )
        except httpx.HTTPError:
            return


def build_tracer(host: str, public_key: str, secret_key: str) -> AskTracer:
    if host and public_key and secret_key:
        return LangfuseTracer(host, public_key, secret_key)
    return MemoryTracer()
