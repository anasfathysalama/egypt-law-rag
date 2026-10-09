"""DVC stage: article JSON to a local vector index."""

from __future__ import annotations

from pathlib import Path

from egylaw_rag.corpus.store import read_articles
from egylaw_rag.index.chunk import chunk_articles
from egylaw_rag.index.embed import LocalEmbedder
from egylaw_rag.index.store import build_index

ROOT = Path(__file__).resolve().parents[1]
ARTICLES_PATH = ROOT / "data" / "processed" / "civil_code.json"
INDEX_PATH = ROOT / "data" / "index"


def main() -> None:
    chunks = chunk_articles(read_articles(ARTICLES_PATH))
    index = build_index(chunks, LocalEmbedder())
    index.save(INDEX_PATH)
    print(f"wrote {len(chunks)} chunks to {INDEX_PATH}")


if __name__ == "__main__":
    main()
