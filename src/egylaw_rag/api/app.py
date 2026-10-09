"""FastAPI app: POST /ask and GET /health."""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field, field_validator
from starlette.responses import Response

from egylaw_rag.api.generate import Generator, MissingApiKey
from egylaw_rag.api.groq import GroqGenerator
from egylaw_rag.api.ollama import OllamaGenerator
from egylaw_rag.api.service import answer_question, stream_answer
from egylaw_rag.config import get_settings
from egylaw_rag.index.embed import Embedder, LocalEmbedder
from egylaw_rag.index.store import VectorIndex
from egylaw_rag.observe.drift import baseline_vector, cosine_similarity
from egylaw_rag.observe.metrics import DRIFT, metrics_payload, record_tokens
from egylaw_rag.observe.trace import AskTracer, build_tracer


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    stream: bool = False

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
    tracer: AskTracer | None = None,
) -> FastAPI:
    app = FastAPI(title="Egyptian Civil Code RAG")
    settings = get_settings()
    state: dict[str, object] = {
        "index": index,
        "embedder": embedder,
        "generator": generator,
        "tracer": tracer
        if tracer is not None
        else build_tracer(
            settings.langfuse_host,
            settings.langfuse_public_key,
            settings.langfuse_secret_key,
        ),
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
            if get_settings().generator_backend == "ollama":
                loaded = OllamaGenerator()
            else:
                loaded = GroqGenerator()
            state["generator"] = loaded
        return loaded  # type: ignore[return-value]

    def current_tracer() -> AskTracer:
        return state["tracer"]  # type: ignore[return-value]

    def record_drift(question: str) -> None:
        index = current_index()
        if len(index.vectors) == 0:
            return
        query = current_embedder().embed_query(question)
        DRIFT.set(cosine_similarity(query, baseline_vector(index.vectors)))

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="healthy", documents_indexed=len(current_index().chunks))

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(content=metrics_payload(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/ask", response_model=None)
    def ask(body: AskRequest) -> AskResponse | StreamingResponse:
        try:
            if body.stream:
                record_drift(body.question)

                def tokens() -> Iterator[str]:
                    parts: list[str] = []
                    for piece in stream_answer(
                        current_index(),
                        current_embedder(),
                        current_generator(),
                        body.question,
                        tracer=current_tracer(),
                    ):
                        parts.append(piece)
                        yield piece
                    record_tokens(body.question, "".join(parts))

                return StreamingResponse(tokens(), media_type="text/plain; charset=utf-8")
            answer, sources = answer_question(
                current_index(),
                current_embedder(),
                current_generator(),
                body.question,
                tracer=current_tracer(),
            )
        except MissingApiKey as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        record_tokens(body.question, answer)
        record_drift(body.question)
        return AskResponse(answer=answer, sources=sources)

    return app


@lru_cache(maxsize=1)
def get_app() -> FastAPI:
    return create_app()


app = get_app()
