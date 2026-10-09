"""One Civil Code article, the unit the rest of the pipeline cites."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ArticleRecord(BaseModel):
    article_number: int
    part: str | None = None
    book: str | None = None
    chapter: str | None = None
    section: str | None = None
    topic: str | None = None
    text_ar: str
    text_en: str
    text_normalized: str
    is_repealed: bool
    source_defect: str | None = None
    source_page: int = Field(ge=1)
    citation: str
