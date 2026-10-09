"""Turn retrieved articles into an answer. Gemini is one implementation of Generator."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Protocol

import httpx

from egylaw_rag.config import get_settings
from egylaw_rag.index.chunk import ArticleChunk


class MissingApiKey(RuntimeError):
    """Raised when /ask needs an API key and that key is empty."""


class Generator(Protocol):
    def generate(self, question: str, chunks: list[ArticleChunk]) -> str: ...


_ARABIC = re.compile(r"[\u0600-\u06FF]")


def question_in_arabic(question: str) -> bool:
    return _ARABIC.search(question) is not None


def build_prompt(question: str, chunks: list[ArticleChunk]) -> str:
    blocks = []
    for chunk in chunks:
        status = "repealed" if chunk.is_repealed else "in force"
        blocks.append(f"{chunk.citation} ({status})\n{chunk.text}")
    articles = "\n\n".join(blocks)
    if question_in_arabic(question):
        language = "The question is in Arabic. Write the entire answer in Arabic."
    else:
        language = "The question is in English. Write the entire answer in English."
    return (
        "Answer the question using only the Egyptian Civil Code articles below. "
        "If they do not contain the answer, say so. "
        "Cite articles as 'Egyptian Civil Code, Article N'. "
        "Do not invent text for a repealed article. "
        f"{language}\n\n"
        f"{articles}\n\nQuestion: {question}"
    )


class GeminiGenerator:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        api_base: str | None = None,
        post: Callable[[str, dict[str, object], dict[str, str]], dict[str, object]] | None = None,
    ) -> None:
        settings = get_settings()
        self.api_key = settings.gemini_api_key if api_key is None else api_key
        self.model = model or settings.gemini_model
        self.api_base = (api_base or settings.gemini_api_base).rstrip("/")
        self._post = post

    def generate(self, question: str, chunks: list[ArticleChunk]) -> str:
        if not self.api_key:
            raise MissingApiKey("Set GEMINI_API_KEY in .env before calling /ask.")
        url = f"{self.api_base}/models/{self.model}:generateContent"
        payload: dict[str, object] = {
            "contents": [{"parts": [{"text": build_prompt(question, chunks)}]}],
        }
        headers = {"x-goog-api-key": self.api_key}
        if self._post is None:
            body = _post_json(url, payload, headers)
        else:
            body = self._post(url, payload, headers)
        return _answer_text(body)


def _post_json(url: str, payload: dict[str, object], headers: dict[str, str]) -> dict[str, object]:
    response = httpx.post(url, json=payload, headers=headers, timeout=60.0)
    if response.status_code >= 400:
        raise RuntimeError(f"Gemini request failed ({response.status_code}): {response.text}")
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("Gemini returned a response that is not an object.")
    return data


def _answer_text(body: dict[str, object]) -> str:
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("Gemini returned no answer.")
    content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list) or not parts or not isinstance(parts[0], dict):
        raise RuntimeError("Gemini returned no answer text.")
    text = parts[0].get("text")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Gemini returned no answer text.")
    return text.strip()
