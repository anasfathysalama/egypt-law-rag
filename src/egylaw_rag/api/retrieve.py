"""Pick the Civil Code chunks that should ground an answer."""

from __future__ import annotations

import re

from egylaw_rag.index.chunk import ArticleChunk
from egylaw_rag.index.embed import Embedder
from egylaw_rag.index.store import VectorIndex, search_text

_AR_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_ARTICLE = re.compile(
    r"(?:articles?|الماد(?:ة|ه)|مادة)\s*[\(\[]?\s*([0-9\u0660-\u0669]+)",
    re.IGNORECASE,
)


def mentioned_articles(question: str) -> list[int]:
    """Article numbers named in the question, in the order they appear."""
    seen: set[int] = set()
    numbers: list[int] = []
    for raw in _ARTICLE.findall(question.translate(_AR_INDIC)):
        number = int(raw)
        if number not in seen:
            seen.add(number)
            numbers.append(number)
    return numbers


def retrieve(
    index: VectorIndex,
    embedder: Embedder,
    question: str,
    k: int = 5,
) -> list[ArticleChunk]:
    """Use the named article when the question cites exactly one. Otherwise search."""
    numbers = mentioned_articles(question)
    if len(numbers) == 1:
        return [chunk for chunk in index.chunks if chunk.article_number == numbers[0]]
    return search_text(index, embedder, question, k)
