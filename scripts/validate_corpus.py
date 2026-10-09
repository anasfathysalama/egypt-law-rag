"""DVC stage: stop the pipeline when the article JSON fails its checks."""

from __future__ import annotations

from pathlib import Path

from egylaw_rag.corpus.store import read_articles
from egylaw_rag.corpus.validate import validate_articles

ROOT = Path(__file__).resolve().parents[1]
ARTICLES_PATH = ROOT / "data" / "processed" / "civil_code.json"


def main() -> None:
    errors = validate_articles(read_articles(ARTICLES_PATH))
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"validated {ARTICLES_PATH}")


if __name__ == "__main__":
    main()
