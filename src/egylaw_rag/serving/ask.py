"""Async /ask used by the BentoML service. The core is the same answer_question."""

from __future__ import annotations

from egylaw_rag.api.generate import Generator
from egylaw_rag.api.service import answer_question
from egylaw_rag.index.embed import Embedder
from egylaw_rag.index.store import VectorIndex


async def ask_async(
    index: VectorIndex,
    embedder: Embedder,
    generator: Generator,
    question: str,
) -> dict[str, str | list[str]]:
    answer, sources = answer_question(index, embedder, generator, question)
    return {"answer": answer, "sources": sources}
