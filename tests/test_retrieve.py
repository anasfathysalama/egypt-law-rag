from egylaw_rag.api.retrieve import mentioned_articles, retrieve
from egylaw_rag.index.chunk import chunk_articles
from egylaw_rag.index.store import build_index
from tests.test_index import _article, _KeywordEmbedder


def _index():
    articles = [
        _article(147, "العقد شريعة المتعاقدين contract"),
        _article(54, "Articles 54-80 have been repealed", repealed=True),
    ]
    return build_index(chunk_articles(articles), _KeywordEmbedder())


def test_mentioned_articles_reads_english_and_arabic() -> None:
    assert mentioned_articles("Article 147") == [147]
    assert mentioned_articles("المادة ١٤٧") == [147]
    assert mentioned_articles("what is a contract") == []


def test_one_named_article_skips_vector_search() -> None:
    hits = retrieve(_index(), _KeywordEmbedder(), "Article 54")
    assert [chunk.article_number for chunk in hits] == [54]
    assert hits[0].is_repealed is True


def test_open_question_uses_vector_search() -> None:
    hits = retrieve(_index(), _KeywordEmbedder(), "contract", k=1)
    assert hits[0].article_number == 147


def test_missing_article_returns_nothing() -> None:
    assert retrieve(_index(), _KeywordEmbedder(), "Article 9999") == []
