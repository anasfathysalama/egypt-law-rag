"""Article chunks, embeddings, and the on-disk vector index."""

from egylaw_rag.index.chunk import ArticleChunk, chunk_articles
from egylaw_rag.index.store import VectorIndex, build_index, search_text

__all__ = ["ArticleChunk", "VectorIndex", "build_index", "chunk_articles", "search_text"]
