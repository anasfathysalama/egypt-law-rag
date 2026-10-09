"""Send five retrieval runs to the MLflow server. Files land in reports/mlflow."""

from __future__ import annotations

from egylaw_rag.config import get_settings
from egylaw_rag.index.embed import LocalEmbedder
from egylaw_rag.index.store import VectorIndex
from egylaw_rag.tracking import QUERIES, log_retrieval_run, top_hit


def main() -> None:
    settings = get_settings()
    index = VectorIndex.load(settings.index)
    embedder = LocalEmbedder(settings.embedding_model)
    for number, query in enumerate(QUERIES, start=1):
        article, citation, latency_ms = top_hit(index, embedder, query)
        log_retrieval_run(
            settings.mlflow_tracking_uri,
            settings.mlflow_experiment,
            settings.embedding_model,
            query,
            article,
            citation,
            latency_ms,
            len(index.chunks),
        )
        print(f"run {number}: article {article} in {latency_ms:.0f} ms")


if __name__ == "__main__":
    main()
