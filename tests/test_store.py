from egylaw_rag.corpus.extract import RawRow
from egylaw_rag.corpus.models import ArticleRecord
from egylaw_rag.corpus.store import read_articles, read_rows, write_articles, write_rows


def test_row_json_roundtrip(tmp_path) -> None:
    path = tmp_path / "rows.json"
    rows = [RawRow(page=2, english="Article 10\nEgyptian law.", arabic="قانون")]
    write_rows(path, rows)
    assert read_rows(path) == rows


def test_article_json_roundtrip(tmp_path) -> None:
    path = tmp_path / "articles.json"
    article = ArticleRecord(
        article_number=54,
        text_ar="ملغاة",
        text_en="Repealed.",
        text_normalized="ملغاة",
        is_repealed=True,
        source_page=7,
        citation="Egyptian Civil Code, Article 54",
    )
    write_articles(path, [article])
    assert read_articles(path) == [article]
