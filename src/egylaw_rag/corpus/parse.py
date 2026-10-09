"""Turn PDF rows into one record per article, including repealed numbers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from egylaw_rag.corpus.arabic import normalize_arabic, repair_arabic, repair_line
from egylaw_rag.corpus.extract import RawRow
from egylaw_rag.corpus.models import ArticleRecord

_ARTICLE = re.compile(r"^Article\s*(\d+)\b", re.IGNORECASE)
_RANGE = re.compile(r"Articles?\s*(\d+)\s*[-–—]\s*(\d+)", re.IGNORECASE)
_REPEAL = re.compile(r"repeal", re.IGNORECASE)
_PART = re.compile(r"^(?:FIRST|SECOND|THIRD|FOURTH|FIFTH)\s+PART\b", re.IGNORECASE)
_BOOK = re.compile(r"^BOOK\b", re.IGNORECASE)
_CHAPTER = re.compile(r"^CHAPTER\b", re.IGNORECASE)
_SECTION = re.compile(r"^SECTION\b", re.IGNORECASE)
_TOPIC = re.compile(r"^\d+\s*[.\-]\s*\S")
_AR_LABEL = re.compile(r"^مادة\s*(?:\(\s*[0-9\u0660-\u0669]+\s*\))?\s*$")
_AR_ARTICLE = re.compile(r"^مادة\s*(?:\(\s*)?([0-9\u0660-\u0669]+)\s*\)?\s*$")
_AR_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


@dataclass
class _Hierarchy:
    part: str | None = None
    book: str | None = None
    chapter: str | None = None
    section: str | None = None
    topic: str | None = None

    def snapshot(self) -> _Hierarchy:
        return _Hierarchy(
            part=self.part,
            book=self.book,
            chapter=self.chapter,
            section=self.section,
            topic=self.topic,
        )


@dataclass
class _Draft:
    number: int
    page: int
    hierarchy: _Hierarchy
    english: list[str] = field(default_factory=list)
    arabic: list[str] = field(default_factory=list)


def _heading_level(english: str) -> str | None:
    stripped = english.strip()
    if not stripped:
        return None
    first = stripped.splitlines()[0].strip()
    if _PART.match(first):
        return "part"
    if _BOOK.match(first):
        return "book"
    if _CHAPTER.match(first):
        return "chapter"
    if _SECTION.match(first):
        return "section"
    if _TOPIC.match(first) and len(first) <= 160:
        return "topic"
    # Short title lines. A continuation sentence has a comma or a final period.
    if (
        len(stripped) <= 120
        and not first[:1].islower()
        and "," not in stripped
        and not stripped.endswith(".")
    ):
        return "topic"
    return None


def _heading_text(english: str) -> str:
    return " ".join(line.strip() for line in english.splitlines() if line.strip())


def _apply_heading(hierarchy: _Hierarchy, level: str, title: str) -> None:
    if level == "part":
        hierarchy.part = title
        hierarchy.book = None
        hierarchy.chapter = None
        hierarchy.section = None
        hierarchy.topic = None
    elif level == "book":
        hierarchy.book = title
        hierarchy.chapter = None
        hierarchy.section = None
        hierarchy.topic = None
    elif level == "chapter":
        hierarchy.chapter = title
        hierarchy.section = None
        hierarchy.topic = None
    elif level == "section":
        hierarchy.section = title
        hierarchy.topic = None
    else:
        hierarchy.topic = title


def _arabic_article_number(arabic: str) -> int | None:
    """Article 452 has no English row. Its number is on the Arabic label line."""
    first = arabic.strip().splitlines()[0] if arabic.strip() else ""
    if not first:
        return None
    match = _AR_ARTICLE.match(repair_line(first))
    if match is None:
        return None
    return int(match.group(1).translate(_AR_INDIC))


def _strip_arabic_label(arabic: str) -> str:
    lines = [line for line in arabic.splitlines() if line.strip()]
    if lines and _AR_LABEL.match(lines[0]):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _source_defect(english: str, arabic: str) -> str | None:
    missing_english = not english.strip()
    missing_arabic = not arabic.strip()
    if missing_english and missing_arabic:
        return "empty_text"
    if missing_arabic:
        return "missing_arabic"
    if missing_english:
        return "missing_english"
    return None


def _record_from_parts(
    number: int,
    page: int,
    hierarchy: _Hierarchy,
    english: str,
    arabic: str,
    *,
    repealed: bool,
) -> ArticleRecord:
    repaired = _strip_arabic_label(repair_arabic(arabic))
    return ArticleRecord(
        article_number=number,
        part=hierarchy.part,
        book=hierarchy.book,
        chapter=hierarchy.chapter,
        section=hierarchy.section,
        topic=hierarchy.topic,
        text_ar=repaired,
        text_en=english.strip(),
        text_normalized=normalize_arabic(repaired),
        is_repealed=repealed,
        source_defect=_source_defect(english, repaired),
        source_page=page,
        citation=f"Egyptian Civil Code, Article {number}",
    )


def _record_from_draft(draft: _Draft) -> ArticleRecord:
    return _record_from_parts(
        draft.number,
        draft.page,
        draft.hierarchy,
        "\n".join(draft.english),
        "\n".join(draft.arabic),
        repealed=False,
    )


def _repeal_span(english: str) -> tuple[int, int] | None:
    match = _RANGE.search(english)
    if match is None or _REPEAL.search(english) is None:
        return None
    start, end = int(match.group(1)), int(match.group(2))
    if end < start:
        start, end = end, start
    return start, end


def parse_rows(rows: list[RawRow]) -> list[ArticleRecord]:
    """Parse table rows into articles  sorted by article number."""
    hierarchy = _Hierarchy()
    draft: _Draft | None = None
    records: list[ArticleRecord] = []

    def flush() -> None:
        nonlocal draft
        if draft is not None:
            records.append(_record_from_draft(draft))
            draft = None

    for row in rows:
        repeal = _repeal_span(row.english)
        if repeal is not None:
            flush()
            start, end = repeal
            for number in range(start, end + 1):
                records.append(
                    _record_from_parts(
                        number,
                        row.page,
                        hierarchy,
                        row.english,
                        row.arabic,
                        repealed=True,
                    )
                )
            continue

        article_match = _ARTICLE.match(row.english.strip())
        arabic_number = None if article_match is not None else _arabic_article_number(row.arabic)
        if arabic_number is not None:
            flush()
            draft = _Draft(number=arabic_number, page=row.page, hierarchy=hierarchy.snapshot())
            if row.english.strip():
                draft.english.append(row.english.strip())
            draft.arabic.append(row.arabic)
            continue

        if article_match is not None:
            number = int(article_match.group(1))
            if draft is not None and draft.number == number:
                body = _ARTICLE.sub("", row.english, count=1).strip()
                if body:
                    draft.english.append(body)
                if row.arabic:
                    draft.arabic.append(row.arabic)
                continue
            flush()
            body_lines = row.english.strip().splitlines()[1:]
            draft = _Draft(
                number=number,
                page=row.page,
                hierarchy=hierarchy.snapshot(),
            )
            body = "\n".join(body_lines).strip()
            if body:
                draft.english.append(body)
            if row.arabic:
                draft.arabic.append(row.arabic)
            continue

        level = _heading_level(row.english)
        if level is not None:
            _apply_heading(hierarchy, level, _heading_text(row.english))
            continue

        if draft is None:
            continue
        if row.english:
            draft.english.append(row.english)
        if row.arabic:
            draft.arabic.append(row.arabic)

    flush()
    records.sort(key=lambda record: record.article_number)
    return records
