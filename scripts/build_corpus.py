"""Write the parsed Civil Code JSON for local inspection.

The full file stays out of git until DVC tracks it. A 20-article fixture is
written for tests.
"""

from __future__ import annotations

import json

from egylaw_rag.config import get_settings
from egylaw_rag.corpus.build import build_corpus
from egylaw_rag.corpus.validate import validate_articles

_settings = get_settings()
PDF_PATH = _settings.pdf
OUTPUT_PATH = _settings.articles
FIXTURE_PATH = _settings.fixture
SAMPLE_NUMBERS = (
    1,
    10,
    54,
    80,
    81,
    147,
    200,
    388,
    389,
    417,
    418,
    452,
    500,
    750,
    900,
    1022,
    1100,
    1147,
    1148,
    1149,
)


def main() -> None:
    articles = build_corpus(PDF_PATH)
    errors = validate_articles(articles)
    if errors:
        raise SystemExit("\n".join(errors))
    payload = [article.model_dump() for article in articles]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    by_number = {article.article_number: article.model_dump() for article in articles}
    sample = [by_number[number] for number in SAMPLE_NUMBERS]
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(payload)} articles to {OUTPUT_PATH}")
    print(f"wrote {len(sample)} sample articles to {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
