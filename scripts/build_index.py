"""DVC stage: article JSON to a local vector index."""

from __future__ import annotations

from egylaw_rag.config import get_settings
from egylaw_rag.corpus.store import read_articles
from egylaw_rag.index.chunk import chunk_articles
from egylaw_rag.index.embed import LocalEmbedder
from egylaw_rag.index.store import build_index

settings = get_settings()
ARTICLES_PATH = settings.articles
INDEX_PATH = settings.index


def main() -> None:
    chunks = chunk_articles(read_articles(ARTICLES_PATH))
    index = build_index(chunks, LocalEmbedder(settings.embedding_model))
    index.save(INDEX_PATH)
    print(f"wrote {len(chunks)} chunks to {INDEX_PATH}")


if __name__ == "__main__":
    main()
