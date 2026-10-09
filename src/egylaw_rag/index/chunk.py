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
    part: str | None = None
    book: str | None = None
    chapter: str | None = None
    section: str | None = None
    paragraph: int | None = None


def _chunk(
    article: ArticleRecord,
    *,
    chunk_id: str,
    text: str,
    paragraph: int | None = None,
) -> ArticleChunk:
    return ArticleChunk(
        chunk_id=chunk_id,
        article_number=article.article_number,
        citation=article.citation,
        text=text,
        is_repealed=article.is_repealed,
        part=article.part,
        book=article.book,
        chapter=article.chapter,
        section=article.section,
        paragraph=paragraph,
    )


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
        return [_chunk(article, chunk_id=str(article.article_number), text=text)]
    return [
        _chunk(
            article,
            chunk_id=f"{article.article_number}-p{index}",
            text=paragraph,
            paragraph=index,
        )
        for index, paragraph in enumerate(paragraphs, start=1)
    ]


def chunk_articles(articles: list[ArticleRecord]) -> list[ArticleChunk]:
    chunks: list[ArticleChunk] = []
    for article in articles:
        chunks.extend(chunk_article(article))
    return chunks


def chunk_windows(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into windows. Each next window repeats `overlap` characters."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    cleaned = text.strip()
    if len(cleaned) <= chunk_size:
        return [cleaned] if cleaned else []
    windows: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        piece = cleaned[start:end].strip()
        if piece:
            windows.append(piece)
        if end == len(cleaned):
            break
        start = end - overlap
    return windows


def chunk_article_config(
    article: ArticleRecord,
    chunk_size: int,
    overlap: int,
) -> list[ArticleChunk]:
    """Window a live article. A repealed article stays one flagged chunk."""
    text = (article.text_normalized or article.text_ar).strip()
    if article.is_repealed:
        return [_chunk(article, chunk_id=str(article.article_number), text=text)]
    parts = chunk_windows(text, chunk_size, overlap)
    if len(parts) <= 1:
        return [_chunk(article, chunk_id=str(article.article_number), text=text)]
    return [
        _chunk(article, chunk_id=f"{article.article_number}-w{index}", text=part)
        for index, part in enumerate(parts, start=1)
    ]


def chunk_articles_config(
    articles: list[ArticleRecord],
    chunk_size: int,
    overlap: int,
) -> list[ArticleChunk]:
    chunks: list[ArticleChunk] = []
    for article in articles:
        chunks.extend(chunk_article_config(article, chunk_size, overlap))
    return chunks
