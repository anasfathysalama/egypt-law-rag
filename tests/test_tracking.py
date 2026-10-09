from egylaw_rag.index.chunk import chunk_articles
from egylaw_rag.index.store import build_index
from egylaw_rag.tracking import (
    QUERIES,
    faithfulness_score,
    log_chunk_run,
    log_retrieval_run,
    top_hit,
)
from tests.test_index import _article, _KeywordEmbedder


def test_faithfulness_is_the_share_of_gold_words_in_context() -> None:
    score = faithfulness_score("the contract binds the parties", ["contract binds"])
    assert score == 0.5
    assert faithfulness_score("contract", ["the contract binds"]) == 1.0


def test_chunk_run_logs_size_overlap_model_and_faithfulness(tmp_path) -> None:
    import mlflow

    tracking_uri = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    log_chunk_run(tracking_uri, "chunking-comparison", 400, 50, "test-model", 0.8)
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name("chunking-comparison")
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    row = runs.iloc[0]
    assert row["params.chunk_size"] == "400"
    assert row["params.overlap"] == "50"
    assert row["params.embedding_model"] == "test-model"
    assert row["metrics.faithfulness"] == 0.8


def test_five_queries_are_logged() -> None:
    assert len(QUERIES) == 5


def test_top_hit_returns_the_matching_article() -> None:
    index = build_index(
        chunk_articles([_article(147, "contract")]),
        _KeywordEmbedder(),
    )
    article, citation, latency_ms = top_hit(index, _KeywordEmbedder(), "contract")
    assert article == 147
    assert citation == "Egyptian Civil Code, Article 147"
    assert latency_ms >= 0


def test_log_retrieval_run_writes_a_local_experiment(tmp_path) -> None:
    import mlflow

    tracking_uri = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    log_retrieval_run(
        tracking_uri,
        "test-retrieval",
        "test-model",
        "contract",
        147,
        "Egyptian Civil Code, Article 147",
        12.5,
        1158,
    )
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name("test-retrieval")
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1
    assert runs.iloc[0]["params.top_citation"] == "Egyptian Civil Code, Article 147"
