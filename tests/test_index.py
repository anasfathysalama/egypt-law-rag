import numpy as np

from egylaw_rag.corpus.models import ArticleRecord
from egylaw_rag.index.chunk import chunk_articles
from egylaw_rag.index.store import build_index, rebuild_index, search_text


class _KeywordEmbedder:
    """Two-axis stand-in so the index test does not download a model."""

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.embed_query(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        contract = 1.0 if "عقد" in text or "contract" in text.lower() else 0.0
        repealed = 1.0 if "ملغاة" in text or "repealed" in text.lower() else 0.0
        return np.asarray([contract, repealed], dtype=np.float32)


def _article(
    number: int,
    text: str,
    *,
    repealed: bool = False,
    part: str | None = None,
    book: str | None = None,
    chapter: str | None = None,
    section: str | None = None,
) -> ArticleRecord:
    return ArticleRecord(
        article_number=number,
        text_ar=text,
        text_en=text,
        text_normalized=text,
        is_repealed=repealed,
        part=part,
        book=book,
        chapter=chapter,
        section=section,
        source_page=1,
        citation=f"Egyptian Civil Code, Article {number}",
    )


def test_search_returns_the_article_citation() -> None:
    articles = [
        _article(147, "العقد شريعة المتعاقدين contract"),
        _article(54, "ملغاة repealed", repealed=True),
    ]
    index = build_index(chunk_articles(articles), _KeywordEmbedder())
    hits = search_text(index, _KeywordEmbedder(), "contract", k=1)
    assert hits[0].citation == "Egyptian Civil Code, Article 147"
    assert hits[0].is_repealed is False


def test_rebuild_includes_one_added_document() -> None:
    embedder = _KeywordEmbedder()
    original = [_article(147, "contract")]
    first = rebuild_index(original, embedder)
    added = _article(54, "repealed", repealed=True)
    second = rebuild_index([*original, added], embedder)
    assert [chunk.article_number for chunk in first.chunks] == [147]
    assert {chunk.article_number for chunk in second.chunks} == {147, 54}
    hits = search_text(second, embedder, "repealed", k=1)
    assert hits[0].article_number == 54
    assert hits[0].is_repealed is True


def test_index_roundtrip(tmp_path) -> None:
    index = build_index(
        chunk_articles([_article(10, "القانون المصري", book="BOOK I")]),
        _KeywordEmbedder(),
    )
    index.save(tmp_path)
    loaded = type(index).load(tmp_path)
    assert loaded.chunks[0].article_number == 10
    assert loaded.chunks[0].book == "BOOK I"
    assert loaded.vectors.shape[0] == 1
