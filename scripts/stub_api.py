"""A local /ask that never calls Groq or Ollama. Locust targets this process."""

from __future__ import annotations

import numpy as np
import uvicorn

from egylaw_rag.api.app import create_app
from egylaw_rag.corpus.models import ArticleRecord
from egylaw_rag.index.chunk import chunk_articles
from egylaw_rag.index.store import build_index
from egylaw_rag.observe.trace import MemoryTracer


class _StubEmbedder:
    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.embed_query(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        contract = 1.0 if "contract" in text.lower() or "عقد" in text else 0.0
        return np.asarray([contract, 1.0], dtype=np.float32)


class _StubGenerator:
    def generate(self, question: str, chunks: list) -> str:
        return f"Stub answer for {chunks[0].citation}."

    def stream(self, question: str, chunks: list):
        yield "Stub "
        yield f"answer for {chunks[0].citation}."


def main() -> None:
    article = ArticleRecord(
        article_number=147,
        text_ar="العقد شريعة المتعاقدين",
        text_en="The contract makes the law of the parties.",
        text_normalized="العقد شريعة المتعاقدين contract",
        is_repealed=False,
        source_page=16,
        citation="Egyptian Civil Code, Article 147",
    )
    embedder = _StubEmbedder()
    index = build_index(chunk_articles([article]), embedder)
    app = create_app(
        index=index,
        embedder=embedder,
        generator=_StubGenerator(),
        tracer=MemoryTracer(),
    )
    uvicorn.run(app, host="127.0.0.1", port=8099)


if __name__ == "__main__":
    main()
