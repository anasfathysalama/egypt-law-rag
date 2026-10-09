"""Decide when to call the model and which citations to return."""

from __future__ import annotations

from collections.abc import Iterator

from egylaw_rag.api.generate import Generator, question_in_arabic
from egylaw_rag.api.retrieve import retrieve
from egylaw_rag.eval.ragas import ragas_faithfulness
from egylaw_rag.guardrails.pii import iter_redacted, redact_pii
from egylaw_rag.index.chunk import ArticleChunk
from egylaw_rag.index.embed import Embedder
from egylaw_rag.index.store import VectorIndex
from egylaw_rag.observe.trace import AskTracer


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
    tracer: AskTracer | None = None,
) -> tuple[str, list[str]]:
    return _finish(index, embedder, generator, question, k, tracer)


def stream_answer(
    index: VectorIndex,
    embedder: Embedder,
    generator: Generator,
    question: str,
    k: int = 5,
    tracer: AskTracer | None = None,
) -> Iterator[str]:
    fixed, chunks = _prepare(index, embedder, question, k, tracer)
    if fixed is not None:
        yield from iter_redacted([fixed])
        return
    if tracer is not None:
        tracer.span("generate")
    pieces: list[str] = []
    for piece in iter_redacted(_token_iter(generator, question, chunks)):
        pieces.append(piece)
        yield piece
    if tracer is not None:
        tracer.score(
            "faithfulness",
            ragas_faithfulness("".join(pieces), [chunk.text for chunk in chunks]),
        )


def _finish(
    index: VectorIndex,
    embedder: Embedder,
    generator: Generator,
    question: str,
    k: int,
    tracer: AskTracer | None,
) -> tuple[str, list[str]]:
    fixed, chunks = _prepare(index, embedder, question, k, tracer)
    if fixed is not None:
        return redact_pii(fixed), citations(chunks) if chunks else []
    if tracer is not None:
        tracer.span("generate")
    answer = redact_pii(generator.generate(question, chunks))
    if tracer is not None:
        tracer.score("faithfulness", ragas_faithfulness(answer, [chunk.text for chunk in chunks]))
    return answer, citations(chunks)


def _prepare(
    index: VectorIndex,
    embedder: Embedder,
    question: str,
    k: int,
    tracer: AskTracer | None,
) -> tuple[str | None, list[ArticleChunk]]:
    if tracer is not None:
        tracer.span("retrieve")
    chunks = retrieve(index, embedder, question, k)
    if not chunks:
        if question_in_arabic(question):
            return "لا توجد مادة في القانون المدني المفهرس تطابق هذا السؤال.", []
        return "No article in the indexed Civil Code matches this question.", []
    if all(chunk.is_repealed for chunk in chunks):
        return _repealed_answer(question, chunks), chunks
    return None, chunks


def _repealed_answer(question: str, chunks: list[ArticleChunk]) -> str:
    numbers = list(dict.fromkeys(chunk.article_number for chunk in chunks))
    note = chunks[0].text.strip()
    listed = ", ".join(str(number) for number in numbers)
    if question_in_arabic(question):
        if len(numbers) == 1:
            return f"المادة {numbers[0]} من القانون المدني المصري ملغاة. {note}"
        return f"المواد {listed} من القانون المدني المصري ملغاة. {note}"
    if len(numbers) == 1:
        return f"Article {numbers[0]} of the Egyptian Civil Code has been repealed. {note}"
    return f"Articles {listed} of the Egyptian Civil Code have been repealed. {note}"


def _token_iter(
    generator: Generator,
    question: str,
    chunks: list[ArticleChunk],
) -> Iterator[str]:
    stream = getattr(generator, "stream", None)
    if stream is None:
        yield generator.generate(question, chunks)
        return
    yield from stream(question, chunks)
