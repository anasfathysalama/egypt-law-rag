"""Decide when to call the model and which citations to return."""

from __future__ import annotations

from egylaw_rag.api.generate import Generator, question_in_arabic
from egylaw_rag.api.retrieve import retrieve
from egylaw_rag.index.chunk import ArticleChunk
from egylaw_rag.index.embed import Embedder
from egylaw_rag.index.store import VectorIndex


def citations(chunks: list[ArticleChunk]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for chunk in chunks:
        if chunk.citation not in seen:
            seen.add(chunk.citation)
            ordered.append(chunk.citation)
    return ordered


def answer_question(
    index: VectorIndex,
    embedder: Embedder,
    generator: Generator,
    question: str,
    k: int = 5,
) -> tuple[str, list[str]]:
    chunks = retrieve(index, embedder, question, k)
    if not chunks:
        if question_in_arabic(question):
            return "لا توجد مادة في القانون المدني المفهرس تطابق هذا السؤال.", []
        return "No article in the indexed Civil Code matches this question.", []
    if all(chunk.is_repealed for chunk in chunks):
        numbers = list(dict.fromkeys(chunk.article_number for chunk in chunks))
        note = chunks[0].text.strip()
        listed = ", ".join(str(number) for number in numbers)
        if question_in_arabic(question):
            if len(numbers) == 1:
                answer = f"المادة {numbers[0]} من القانون المدني المصري ملغاة. {note}"
            else:
                answer = f"المواد {listed} من القانون المدني المصري ملغاة. {note}"
        elif len(numbers) == 1:
            answer = f"Article {numbers[0]} of the Egyptian Civil Code has been repealed. {note}"
        else:
            answer = f"Articles {listed} of the Egyptian Civil Code have been repealed. {note}"
        return answer, citations(chunks)
    return generator.generate(question, chunks), citations(chunks)
