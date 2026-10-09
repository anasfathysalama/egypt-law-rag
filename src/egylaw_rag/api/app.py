"""FastAPI app: POST /ask and GET /health."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from egylaw_rag.api.generate import Generator, MissingApiKey
from egylaw_rag.api.groq import GroqGenerator
from egylaw_rag.api.service import answer_question
from egylaw_rag.config import get_settings
from egylaw_rag.index.embed import Embedder, LocalEmbedder
from egylaw_rag.index.store import VectorIndex


class AskRequest(BaseModel):
    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be empty")
        return cleaned


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int


def create_app(
    index: VectorIndex | None = None,
    embedder: Embedder | None = None,
    generator: Generator | None = None,
) -> FastAPI:
    app = FastAPI(title="Egyptian Civil Code RAG")
    state: dict[str, object] = {
        "index": index,
        "embedder": embedder,
        "generator": generator,
    }

    def current_index() -> VectorIndex:
        loaded = state["index"]
        if loaded is None:
            loaded = VectorIndex.load(get_settings().index)
            state["index"] = loaded
        return loaded  # type: ignore[return-value]

    def current_embedder() -> Embedder:
        loaded = state["embedder"]
        if loaded is None:
            loaded = LocalEmbedder()
            state["embedder"] = loaded
        return loaded  # type: ignore[return-value]

    def current_generator() -> Generator:
        loaded = state["generator"]
        if loaded is None:
            loaded = GroqGenerator()
            state["generator"] = loaded
        return loaded  # type: ignore[return-value]

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="healthy", documents_indexed=len(current_index().chunks))

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest) -> AskResponse:
        try:
            answer, sources = answer_question(
                current_index(),
                current_embedder(),
                current_generator(),
                body.question,
            )
        except MissingApiKey as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return AskResponse(answer=answer, sources=sources)

    return app


@lru_cache(maxsize=1)
def get_app() -> FastAPI:
    return create_app()


app = get_app()
