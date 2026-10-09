"""Compare chunk size and overlap, then register the best one in MLflow."""

from __future__ import annotations

from egylaw_rag.config import get_settings
from egylaw_rag.corpus.store import read_articles
from egylaw_rag.index.chunk import chunk_articles_config
from egylaw_rag.index.embed import LocalEmbedder
from egylaw_rag.index.store import build_index
from egylaw_rag.tracking import (
    CHUNKING_EXPERIMENT,
    TEXT_FIELD,
    faithfulness_score,
    log_chunk_run,
    register_best_chunking,
)

# Five setups. Repealed articles stay one chunk inside chunk_articles_config.
CONFIGS = (
    (200, 0),
    (400, 50),
    (700, 0),
    (700, 100),
    (1200, 150),
)
EVAL_ARTICLES = (1, 6, 147, 200, 500)


def _eval_articles(articles: list) -> list:
    by_number = {article.article_number: article for article in articles}
    chosen = [by_number[number] for number in EVAL_ARTICLES if number in by_number]
    chosen_ids = {article.article_number for article in chosen}
    distractors = [article for article in articles if article.article_number not in chosen_ids]
    return chosen + distractors[:40]


def _query(article) -> str:
    text = (article.text_en or article.text_normalized).strip()
    sentence = text.split(".")[0].strip()
    return sentence[:180] or text[:180]


def _score(index, embedder, questions: list[tuple[str, str]]) -> float:
    scores = []
    for query, gold in questions:
        hits = index.search(embedder.embed_query(query), k=8)
        scores.append(faithfulness_score(gold, [hit.text for hit in hits]))
    return sum(scores) / len(scores)


def main() -> None:
    settings = get_settings()
    articles = read_articles(settings.articles)
    sample = _eval_articles(articles)
    eval_count = len(EVAL_ARTICLES)
    questions = [(_query(article), article.text_normalized) for article in sample[:eval_count]]
    if len(questions) < 5:
        raise RuntimeError("The corpus is missing one of the five evaluation articles.")
    embedder = LocalEmbedder(settings.embedding_model)
    results: list[tuple[int, int, float, str]] = []
    for chunk_size, overlap in CONFIGS:
        chunks = chunk_articles_config(sample, chunk_size, overlap)
        index = build_index(chunks, embedder)
        score = _score(index, embedder, questions)
        run_id = log_chunk_run(
            settings.mlflow_tracking_uri,
            CHUNKING_EXPERIMENT,
            chunk_size,
            overlap,
            settings.embedding_model,
            score,
            TEXT_FIELD,
        )
        results.append((chunk_size, overlap, score, run_id))
        print(f"chunk_size={chunk_size} overlap={overlap} faithfulness={score:.3f}")
    chunk_size, overlap, score, run_id = max(results, key=lambda item: item[2])
    name = register_best_chunking(
        settings.mlflow_tracking_uri,
        run_id,
        chunk_size,
        overlap,
        settings.embedding_model,
        score,
    )
    print(
        f"registered {name}: chunk_size={chunk_size} overlap={overlap} "
        f"faithfulness={score:.3f}"
    )


if __name__ == "__main__":
    main()
