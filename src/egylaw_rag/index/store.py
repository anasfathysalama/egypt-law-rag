"""Save chunk vectors and find the nearest ones for a query."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from egylaw_rag.corpus.models import ArticleRecord
from egylaw_rag.index.chunk import ArticleChunk, chunk_articles
from egylaw_rag.index.embed import Embedder


class VectorIndex:
    def __init__(self, chunks: list[ArticleChunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("each chunk needs one vector")
        self.chunks = chunks
        self.vectors = _normalize(vectors)

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[ArticleChunk]:
        query = _normalize(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
        scores = self.vectors @ query
        order = np.argsort(scores)[::-1][:k]
        return [self.chunks[int(index)] for index in order]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = [chunk.model_dump() for chunk in self.chunks]
        (directory / "chunks.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        np.save(directory / "vectors.npy", self.vectors)

    @classmethod
    def load(cls, directory: Path) -> VectorIndex:
        payload = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        chunks = [ArticleChunk.model_validate(item) for item in payload]
        vectors = np.load(directory / "vectors.npy")
        return cls(chunks, vectors)


def build_index(chunks: list[ArticleChunk], embedder: Embedder) -> VectorIndex:
    vectors = embedder.embed_passages([chunk.text for chunk in chunks])
    return VectorIndex(chunks, vectors)


def search_text(
    index: VectorIndex,
    embedder: Embedder,
    query: str,
    k: int = 5,
) -> list[ArticleChunk]:
    return index.search(embedder.embed_query(query), k)


def rebuild_index(articles: list[ArticleRecord], embedder: Embedder) -> VectorIndex:
    """Embed every article again, including one that was just added."""
    return build_index(chunk_articles(articles), embedder)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (vectors / norms).astype(np.float32)
