import numpy as np

from egylaw_rag.corpus.models import ArticleRecord
from egylaw_rag.index.chunk import chunk_articles
from egylaw_rag.index.store import build_index
from egylaw_rag.serving.ask import ask_async


class _Embedder:
    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.ones(2, dtype=np.float32)


class _Generator:
    def generate(self, question: str, chunks: list) -> str:
        return "The contract binds the parties."


def test_async_ask_returns_the_citation() -> None:
    import asyncio

    article = ArticleRecord(
        article_number=147,
        text_ar="contract",
        text_en="contract",
        text_normalized="contract",
        is_repealed=False,
        source_page=1,
        citation="Egyptian Civil Code, Article 147",
    )
    embedder = _Embedder()
    index = build_index(chunk_articles([article]), embedder)
    result = asyncio.run(ask_async(index, embedder, _Generator(), "contract"))
    assert result["sources"] == ["Egyptian Civil Code, Article 147"]
    assert result["answer"] == "The contract binds the parties."
