"""Turn chunk text into vectors. Tests can pass any object with these two methods."""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np

from egylaw_rag.config import get_settings


class Embedder(Protocol):
    def embed_passages(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


# fastembed has no multilingual-e5-small or bge-m3. These two fit a 16 GB CPU.
PRIMARY_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SECOND_EMBEDDING_MODEL = "minishlab/potion-multilingual-128M"
EMBEDDING_MODELS = (PRIMARY_EMBEDDING_MODEL, SECOND_EMBEDDING_MODEL)


class LocalEmbedder:
    """CPU embedder. The model is downloaded on first use and then reused."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or get_settings().embedding_model
        self._model: object | None = None

    def _embedding_model(self) -> object:
        if self._model is None:
            from fastembed import TextEmbedding

            cache = get_settings().embedding_cache_dir
            cache.mkdir(parents=True, exist_ok=True)
            os.environ["FASTEMBED_CACHE_PATH"] = str(cache)
            self._model = TextEmbedding(self.model_name)
        return self._model

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        vectors = list(self._embedding_model().embed(texts))  # type: ignore[attr-defined]
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vectors = list(self._embedding_model().embed([text]))  # type: ignore[attr-defined]
        return np.asarray(vectors[0], dtype=np.float32)
