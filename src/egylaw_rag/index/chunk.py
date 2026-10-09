"""Split the Civil Code into embeddable pieces without cutting a provision in half."""

from __future__ import annotations

import re

from pydantic import BaseModel

from egylaw_rag.corpus.models import ArticleRecord

# A short article stays whole. Numbered paragraphs split only after this length.
MIN_CHARS_TO_SPLIT = 700
_PARAGRAPH = re.compile(r"(?=\([0-9\u0660-\u0669]+\))")
_PARAGRAPH_START = re.compile(r"^\([0-9\u0660-\u0669]+\)")


class ArticleChunk(BaseModel):
    chunk_id: str
    article_number: int
    citation: str
    text: str
    is_repealed: bool
    paragraph: int | None = None


def _numbered_paragraphs(text: str) -> list[str]:
    parts = [part.strip() for part in _PARAGRAPH.split(text) if part.strip()]
    if parts and not _PARAGRAPH_START.match(parts[0]) and len(parts) > 1:
        parts[1] = f"{parts[0]}\n{parts[1]}"
        parts = parts[1:]
    numbered = [part for part in parts if _PARAGRAPH_START.match(part)]
    if len(numbered) < 2:
        return []
    return numbered


def chunk_article(article: ArticleRecord) -> list[ArticleChunk]:
    """Return one chunk, or one chunk per numbered paragraph when the article is long."""
    text = (article.text_normalized or article.text_ar).strip()
    paragraphs = [] if article.is_repealed else _numbered_paragraphs(text)
    if article.is_repealed or len(text) < MIN_CHARS_TO_SPLIT or not paragraphs:
        return [
            ArticleChunk(
                chunk_id=str(article.article_number),
                article_number=article.article_number,
                citation=article.citation,
                text=text,
                is_repealed=article.is_repealed,
            )
        ]
    return [
        ArticleChunk(
            chunk_id=f"{article.article_number}-p{index}",
            article_number=article.article_number,
            citation=article.citation,
            text=paragraph,
            is_repealed=False,
            paragraph=index,
        )
        for index, paragraph in enumerate(paragraphs, start=1)
    ]


def chunk_articles(articles: list[ArticleRecord]) -> list[ArticleChunk]:
    chunks: list[ArticleChunk] = []
    for article in articles:
        chunks.extend(chunk_article(article))
    return chunks
