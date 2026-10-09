"""Answer from retrieved articles with a Qwen model hosted by Groq."""

from __future__ import annotations

import re
from collections.abc import Callable

import httpx

from egylaw_rag.api.generate import MissingApiKey, build_prompt
from egylaw_rag.config import get_settings
from egylaw_rag.index.chunk import ArticleChunk

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class GroqGenerator:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        api_base: str | None = None,
        max_tokens: int | None = None,
        post: Callable[[str, dict[str, object], dict[str, str]], dict[str, object]] | None = None,
    ) -> None:
        settings = get_settings()
        self.api_key = settings.groq_api_key if api_key is None else api_key
        self.model = model or settings.groq_model
        self.api_base = (api_base or settings.groq_api_base).rstrip("/")
        if max_tokens is None:
            self.max_tokens = settings.groq_max_tokens
        else:
            self.max_tokens = max_tokens
        self._post = post

    def generate(self, question: str, chunks: list[ArticleChunk]) -> str:
        if not self.api_key:
            raise MissingApiKey("Set GROQ_API_KEY in .env before calling /ask.")
        url = f"{self.api_base}/chat/completions"
        payload: dict[str, object] = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "reasoning_effort": "none",
            "messages": [{"role": "user", "content": build_prompt(question, chunks)}],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self._post is None:
            body = _post_json(url, payload, headers)
        else:
            body = self._post(url, payload, headers)
        return _answer_text(body)


def _post_json(url: str, payload: dict[str, object], headers: dict[str, str]) -> dict[str, object]:
    response = httpx.post(url, json=payload, headers=headers, timeout=60.0)
    if response.status_code >= 400:
        raise RuntimeError(f"Groq request failed ({response.status_code}): {response.text}")
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("Groq returned a response that is not an object.")
    return data


def _answer_text(body: dict[str, object]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("Groq returned no answer.")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError("Groq returned no answer.")
    text = message.get("content")
    if not isinstance(text, str):
        raise RuntimeError("Groq returned no answer text.")
    cleaned = _THINK.sub("", text).strip()
    if not cleaned:
        raise RuntimeError("Groq returned no answer text.")
    return cleaned
