from egylaw_rag.corpus.models import ArticleRecord
from egylaw_rag.index.chunk import chunk_article, chunk_article_config, chunk_windows


def _article(**overrides: object) -> ArticleRecord:
    values: dict[str, object] = {
        "article_number": 147,
        "text_ar": "العقد شريعة المتعاقدين",
        "text_en": "The contract makes the law of the parties.",
        "text_normalized": "العقد شريعة المتعاقدين",
        "is_repealed": False,
        "source_page": 20,
        "citation": "Egyptian Civil Code, Article 147",
    }
    values.update(overrides)
    return ArticleRecord.model_validate(values)


def test_short_article_is_one_chunk() -> None:
    chunks = chunk_article(_article())
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "147"
    assert chunks[0].citation == "Egyptian Civil Code, Article 147"
    assert chunks[0].article_number == 147


def test_repealed_article_stays_one_flagged_chunk() -> None:
    long_note = "(١) " + ("ملغاة " * 200) + "(٢) " + ("ملغاة " * 200)
    chunks = chunk_article(
        _article(
            article_number=54,
            text_normalized=long_note,
            is_repealed=True,
            citation="Egyptian Civil Code, Article 54",
        )
    )
    assert len(chunks) == 1
    assert chunks[0].is_repealed is True
    assert chunks[0].citation == "Egyptian Civil Code, Article 54"
    assert chunks[0].article_number == 54


def test_long_article_splits_by_numbered_paragraph() -> None:
    first = "(١) " + ("يلتزم المدين بالوفاء. " * 40)
    second = "(٢) " + ("يجوز للدائن طلب التنفيذ. " * 40)
    chunks = chunk_article(_article(text_normalized=f"{first}\n{second}"))
    assert [chunk.paragraph for chunk in chunks] == [1, 2]
    assert {chunk.citation for chunk in chunks} == {"Egyptian Civil Code, Article 147"}
    assert {chunk.article_number for chunk in chunks} == {147}
    assert chunks[0].chunk_id == "147-p1"


def test_windows_repeat_the_overlap() -> None:
    windows = chunk_windows("abcdefghijklmnopqrstuvwxyz", chunk_size=10, overlap=3)
    assert windows[0] == "abcdefghij"
    assert windows[1].startswith("hij")


def test_repealed_article_is_not_windowed() -> None:
    chunks = chunk_article_config(
        _article(text_normalized="ملغاة " * 80, is_repealed=True),
        chunk_size=20,
        overlap=5,
    )
    assert len(chunks) == 1
    assert chunks[0].is_repealed is True
