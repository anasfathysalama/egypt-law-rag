from egylaw_rag.corpus.extract import RawRow
from egylaw_rag.corpus.parse import parse_rows


def test_parse_rows_expands_repeal_range_and_keeps_hierarchy() -> None:
    rows = [
        RawRow(page=1, english="FIRST PART\nPersons", arabic=""),
        RawRow(page=1, english="BOOK I\nObligations", arabic=""),
        RawRow(page=2, english="Article 53\nA living person has a name.", arabic="اسم"),
        RawRow(
            page=3,
            english="Article 54\n* Articles 54-80 have been repealed by Presidential Decree.",
            arabic="ملغاة",
        ),
        RawRow(page=4, english="Article 81\nA person who has reached majority.", arabic="رشد"),
    ]
    articles = parse_rows(rows)
    by_number = {article.article_number: article for article in articles}

    assert set(by_number) == set(range(53, 82))
    assert by_number[53].is_repealed is False
    assert by_number[53].text_en == "A living person has a name."
    assert by_number[53].part == "FIRST PART Persons"
    assert by_number[53].book == "BOOK I Obligations"
    assert by_number[54].is_repealed is True
    assert by_number[80].is_repealed is True
    assert "repealed" in by_number[80].text_en
    assert by_number[80].text_ar
    assert by_number[81].is_repealed is False
    assert by_number[54].citation == "Egyptian Civil Code, Article 54"


def test_parse_rows_reads_arabic_only_article_label() -> None:
    rows = [
        RawRow(page=5, english="", arabic="(٤٥٢ ( ةدام\nصن"),
    ]
    article = parse_rows(rows)[0]
    assert article.article_number == 452
    assert article.source_defect == "missing_english"
    assert "نص" in article.text_ar


def test_parse_rows_appends_continuation_and_flags_empty_article() -> None:
    rows = [
        RawRow(page=1, english="Article 6\nCapacity begins at birth.", arabic="اهلية"),
        RawRow(page=2, english="When a person loses capacity, prior acts stand.", arabic=""),
        RawRow(page=9, english="Article1022", arabic=""),
    ]
    articles = parse_rows(rows)
    article_6, article_1022 = articles

    assert "prior acts stand" in article_6.text_en
    assert article_1022.article_number == 1022
    assert article_1022.source_defect == "empty_text"
    assert article_1022.text_ar == ""
