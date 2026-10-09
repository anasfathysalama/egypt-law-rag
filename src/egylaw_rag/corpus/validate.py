"""Checks that stop a bad parse before any embedding step."""

from __future__ import annotations

from egylaw_rag.corpus.models import ArticleRecord

EXPECTED_COUNT = 1149
EXPECTED_REPEALED = set(range(54, 81)) | set(range(389, 418))
MAX_TEXT_CHARS = 20_000
# Filled when a real row has a blank side. This PDF's article 452 has both languages.
KNOWN_SOURCE_DEFECTS: dict[int, str] = {}


def validate_articles(articles: list[ArticleRecord]) -> list[str]:
    errors: list[str] = []
    numbers = [article.article_number for article in articles]
    if len(articles) != EXPECTED_COUNT:
        errors.append(f"expected {EXPECTED_COUNT} records, found {len(articles)}")
    if len(numbers) != len(set(numbers)):
        errors.append("duplicate article numbers")
    missing = [number for number in range(1, EXPECTED_COUNT + 1) if number not in set(numbers)]
    if missing:
        errors.append(f"missing article numbers: {missing[:12]}")

    repealed = {article.article_number for article in articles if article.is_repealed}
    if repealed != EXPECTED_REPEALED:
        unexpected = sorted(repealed - EXPECTED_REPEALED)
        absent = sorted(EXPECTED_REPEALED - repealed)
        errors.append(f"repeal mismatch unexpected={unexpected[:8]} absent={absent[:8]}")

    live = EXPECTED_COUNT - len(EXPECTED_REPEALED)
    found_live = sum(1 for article in articles if not article.is_repealed)
    if found_live != live:
        errors.append(f"expected {live} live articles, found {found_live}")

    for article in articles:
        if not isinstance(article.article_number, int):
            errors.append(f"{article.article_number} is not an int")
        if article.citation != f"Egyptian Civil Code, Article {article.article_number}":
            errors.append(f"bad citation for {article.article_number}")
        arabic_len = len(article.text_ar)
        english_len = len(article.text_en)
        if arabic_len > MAX_TEXT_CHARS or english_len > MAX_TEXT_CHARS:
            errors.append(f"article {article.article_number} exceeds {MAX_TEXT_CHARS} characters")
        if not article.text_ar.strip() and article.source_defect is None:
            errors.append(f"article {article.article_number} has empty Arabic")
        expected_defect = KNOWN_SOURCE_DEFECTS.get(article.article_number)
        if article.source_defect != expected_defect:
            errors.append(
                f"article {article.article_number} source_defect {article.source_defect!r}"
            )
        if article.is_repealed and not article.text_ar.strip():
            errors.append(f"repealed article {article.article_number} has no repeal note")
    return errors
