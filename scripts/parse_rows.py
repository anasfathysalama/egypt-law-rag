"""DVC stage: raw rows to one JSON record per article."""

from __future__ import annotations

from egylaw_rag.config import get_settings
from egylaw_rag.corpus.parse import parse_rows
from egylaw_rag.corpus.store import read_rows, write_articles

ROWS_PATH = get_settings().raw_rows
ARTICLES_PATH = get_settings().articles


def main() -> None:
    articles = parse_rows(read_rows(ROWS_PATH))
    write_articles(ARTICLES_PATH, articles)
    print(f"wrote {len(articles)} articles to {ARTICLES_PATH}")


if __name__ == "__main__":
    main()
