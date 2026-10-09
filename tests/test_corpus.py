from pathlib import Path

import pytest

from egylaw_rag.corpus.build import build_corpus
from egylaw_rag.corpus.validate import EXPECTED_COUNT, validate_articles

PDF_PATH = Path("data/raw/law.pdf")


@pytest.fixture(scope="module")
def articles():
    if not PDF_PATH.exists():
        pytest.skip("Civil Code PDF is local raw data")
    return build_corpus(PDF_PATH)


def test_law_pdf_validates(articles) -> None:
    errors = validate_articles(articles)
    assert errors == []
    assert len(articles) == EXPECTED_COUNT


def test_law_pdf_repairs_known_articles(articles) -> None:
    by_number = {article.article_number: article for article in articles}
    article_1 = by_number[1]
    assert article_1.article_number == 1
    assert "Provisions of laws" in article_1.text_en
    assert "النصوص التشريعية" in article_1.text_ar
    assert article_1.text_normalized
    assert article_1.is_repealed is False

    assert "legal capacity" in by_number[6].text_en
    assert by_number[54].is_repealed is True
    assert by_number[417].is_repealed is True
    assert by_number[418].is_repealed is False
    assert by_number[452].text_ar
    assert by_number[452].text_en
    assert by_number[452].source_defect is None
    assert by_number[1022].text_ar
    assert by_number[1022].source_defect is None
    assert "Article 1149" in by_number[1149].citation
