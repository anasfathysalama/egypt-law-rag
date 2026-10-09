"""BentoML service. Run it with: bentoml serve egylaw_rag.serving.service:CivilCodeRag"""

from __future__ import annotations

import bentoml

from egylaw_rag.api.groq import GroqGenerator
from egylaw_rag.api.ollama import OllamaGenerator
from egylaw_rag.config import get_settings
from egylaw_rag.index.embed import LocalEmbedder
from egylaw_rag.index.store import VectorIndex
from egylaw_rag.serving.ask import ask_async


def _generator() -> GroqGenerator | OllamaGenerator:
    settings = get_settings()
    if settings.generator_backend == "ollama":
        return OllamaGenerator()
    return GroqGenerator()


@bentoml.service(name="egylaw_rag")
class CivilCodeRag:
    @bentoml.api
    async def ask(self, question: str) -> dict[str, object]:
        settings = get_settings()
        result = await ask_async(
            VectorIndex.load(settings.index),
            LocalEmbedder(settings.embedding_model),
            _generator(),
            question,
        )
        return {"answer": result["answer"], "sources": result["sources"]}
