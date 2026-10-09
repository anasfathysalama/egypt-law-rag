"""DVC stage: raw rows to one JSON record per article."""

from __future__ import annotations

from pathlib import Path

from egylaw_rag.corpus.parse import parse_rows
from egylaw_rag.corpus.store import read_rows, write_articles

ROOT = Path(__file__).resolve().parents[1]
ROWS_PATH = ROOT / "data" / "interim" / "raw_rows.json"
ARTICLES_PATH = ROOT / "data" / "processed" / "civil_code.json"


def main() -> None:
    articles = parse_rows(read_rows(ROWS_PATH))
    write_articles(ARTICLES_PATH, articles)
    print(f"wrote {len(articles)} articles to {ARTICLES_PATH}")


if __name__ == "__main__":
    main()
