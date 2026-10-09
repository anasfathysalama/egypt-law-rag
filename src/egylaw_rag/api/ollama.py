"""Answer with a small Qwen model served by Ollama on this CPU.

Ollama replaces vLLM. The model name is qwen2.5:0.5b. Its files stay in the
Compose volume on this project disk, not on C:.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator

import httpx

from egylaw_rag.api.generate import build_prompt
from egylaw_rag.config import get_settings
from egylaw_rag.index.chunk import ArticleChunk

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class OllamaGenerator:
    def __init__(
        self,
        model: str | None = None,
        api_base: str | None = None,
        post: Callable[[str, dict[str, object], dict[str, str]], dict[str, object]] | None = None,
        stream_lines: Callable[[str, dict[str, object]], Iterator[str]] | None = None,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.ollama_model
        self.api_base = (api_base or settings.ollama_api_base).rstrip("/")
        self._post = post
        self._stream_lines = stream_lines

    def generate(self, question: str, chunks: list[ArticleChunk]) -> str:
        body = self._complete(question, chunks, stream=False)
        if not isinstance(body, dict):
            raise RuntimeError("Ollama returned a response that is not an object.")
        return _answer_text(body)

    def stream(self, question: str, chunks: list[ArticleChunk]) -> Iterator[str]:
        payload = self._payload(question, chunks, stream=True)
        url = f"{self.api_base}/chat/completions"
        lines = (
            self._stream_lines(url, payload)
            if self._stream_lines is not None
            else _stream_lines(url, payload)
        )
        for line in lines:
            token = _delta_text(line)
            if token:
                yield token

    def _complete(
        self,
        question: str,
        chunks: list[ArticleChunk],
        *,
        stream: bool,
    ) -> dict[str, object]:
        payload = self._payload(question, chunks, stream=stream)
        url = f"{self.api_base}/chat/completions"
        if self._post is None:
            return _post_json(url, payload)
        return self._post(url, payload, {})

    def _payload(
        self,
        question: str,
        chunks: list[ArticleChunk],
        *,
        stream: bool,
    ) -> dict[str, object]:
        return {
            "model": self.model,
            "temperature": 0,
            "stream": stream,
            "messages": [{"role": "user", "content": build_prompt(question, chunks)}],
        }


def _post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    response = httpx.post(url, json=payload, timeout=120.0)
    if response.status_code >= 400:
        raise RuntimeError(f"Ollama request failed ({response.status_code}): {response.text}")
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("Ollama returned a response that is not an object.")
    return data


def _stream_lines(url: str, payload: dict[str, object]) -> Iterator[str]:
    with httpx.stream("POST", url, json=payload, timeout=120.0) as response:
        if response.status_code >= 400:
            detail = response.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed ({response.status_code}): {detail}")
        for line in response.iter_lines():
            if line:
                yield line


def _answer_text(body: dict[str, object]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("Ollama returned no answer.")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError("Ollama returned no answer.")
    text = message.get("content")
    if not isinstance(text, str) or not _THINK.sub("", text).strip():
        raise RuntimeError("Ollama returned no answer text.")
    return _THINK.sub("", text).strip()


def _delta_text(line: str) -> str:
    if not line.startswith("data:"):
        return ""
    data = line.removeprefix("data:").strip()
    if not data or data == "[DONE]":
        return ""
    body = json.loads(data)
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    delta = choices[0].get("delta")
    if not isinstance(delta, dict):
        return ""
    text = delta.get("content")
    return text if isinstance(text, str) else ""
