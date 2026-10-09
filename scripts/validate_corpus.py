"""DVC stage: stop the pipeline when the article JSON fails its checks."""

from __future__ import annotations

from egylaw_rag.config import get_settings
from egylaw_rag.corpus.store import read_articles
from egylaw_rag.corpus.validate import validate_articles

ARTICLES_PATH = get_settings().articles


def main() -> None:
    errors = validate_articles(read_articles(ARTICLES_PATH))
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"validated {ARTICLES_PATH}")


if __name__ == "__main__":
    main()
