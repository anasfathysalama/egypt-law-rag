"""Log retrieval runs to the local MLflow server."""

from __future__ import annotations

import re
import time

import mlflow.pyfunc

from egylaw_rag.api.retrieve import retrieve
from egylaw_rag.index.embed import Embedder
from egylaw_rag.index.store import VectorIndex

CHUNKING_EXPERIMENT = "chunking-comparison"
REGISTRY_NAME = "best-chunking-config"
_TOKEN = re.compile(r"[\w\u0600-\u06FF]{3,}", re.UNICODE)

QUERIES = (
    "contract makes the law of the parties",
    "المادة 147",
    "legal capacity",
    "Article 1",
    "obligation to deliver",
)


def top_hit(
    index: VectorIndex,
    embedder: Embedder,
    query: str,
    k: int = 5,
) -> tuple[int, str, float]:
    started = time.perf_counter()
    hits = retrieve(index, embedder, query, k)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if not hits:
        return 0, "", elapsed_ms
    return hits[0].article_number, hits[0].citation, elapsed_ms


def log_retrieval_run(
    tracking_uri: str,
    experiment: str,
    embedding_model: str,
    query: str,
    top_article: int,
    top_citation: str,
    latency_ms: float,
    documents_indexed: int,
) -> None:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        mlflow.log_param("embedding_model", embedding_model)
        mlflow.log_param("query", query)
        mlflow.log_param("top_citation", top_citation)
        mlflow.log_metric("top_article", top_article)
        mlflow.log_metric("latency_ms", latency_ms)
        mlflow.log_metric("documents_indexed", documents_indexed)


def faithfulness_score(gold: str, contexts: list[str]) -> float:
    """Share of gold-answer words that appear in the retrieved chunks."""
    gold_tokens = set(_TOKEN.findall(gold.lower()))
    if not gold_tokens:
        return 0.0
    context_tokens = set(_TOKEN.findall(" ".join(contexts).lower()))
    return len(gold_tokens & context_tokens) / len(gold_tokens)


def log_chunk_run(
    tracking_uri: str,
    experiment: str,
    chunk_size: int,
    overlap: int,
    embedding_model: str,
    faithfulness: float,
) -> str:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run() as run:
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("overlap", overlap)
        mlflow.log_param("embedding_model", embedding_model)
        mlflow.log_metric("faithfulness", faithfulness)
        return run.info.run_id


class ChunkConfigModel(mlflow.pyfunc.PythonModel):
    """Winning chunk settings. Registered as an MLflow pyfunc model."""

    def __init__(
        self,
        chunk_size: int,
        overlap: int,
        embedding_model: str,
        faithfulness: float,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embedding_model = embedding_model
        self.faithfulness = faithfulness

    def predict(
        self,
        context: object,
        model_input: object,
        params: object = None,
    ) -> dict[str, object]:
        return {
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "embedding_model": self.embedding_model,
            "faithfulness": self.faithfulness,
        }


def register_best_chunking(
    tracking_uri: str,
    run_id: str,
    chunk_size: int,
    overlap: int,
    embedding_model: str,
    faithfulness: float,
) -> str:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    model = ChunkConfigModel(chunk_size, overlap, embedding_model, faithfulness)
    with mlflow.start_run(run_id=run_id):
        info = mlflow.pyfunc.log_model(
            name="chunking-config",
            python_model=model,
            registered_model_name=REGISTRY_NAME,
            pip_requirements=["mlflow"],
        )
    return info.registered_model_version.name if info.registered_model_version else REGISTRY_NAME
