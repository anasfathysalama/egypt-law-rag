"""Score the CI questions and stop the build when faithfulness is under 0.75."""

from __future__ import annotations

import httpx

from egylaw_rag.config import get_settings
from egylaw_rag.corpus.store import read_articles
from egylaw_rag.eval.metrics import (
    FAITHFULNESS_MINIMUM,
    RETRIEVAL_K,
    first_relevant_rank,
    hit_at_k,
    load_questions,
    mean_reciprocal_rank,
    passes_faithfulness,
)
from egylaw_rag.eval.ragas import mean_scores, notify_low_faithfulness, score_example
from egylaw_rag.index.embed import LocalEmbedder
from egylaw_rag.index.store import VectorIndex


def main() -> None:
    settings = get_settings()
    articles = {article.article_number: article for article in read_articles(settings.articles)}
    questions = load_questions(settings.questions_ci)
    missing = [
        question.article_number
        for question in questions
        if question.article_number not in articles
    ]
    if missing:
        raise SystemExit(f"corpus is missing evaluation articles: {missing}")
    index = VectorIndex.load(settings.index)
    embedder = LocalEmbedder(settings.embedding_model)
    rows = []
    ranks: list[int | None] = []
    for question in questions:
        article = articles[question.article_number]
        hits = index.search(embedder.embed_query(question.question_ar), k=RETRIEVAL_K)
        contexts = [hit.text for hit in hits]
        numbers = [hit.article_number for hit in hits]
        rows.append(
            score_example(
                question.question_ar,
                article.text_normalized,
                contexts,
                numbers,
                article.article_number,
                article.text_normalized,
            )
        )
        ranks.append(first_relevant_rank(numbers, question.article_number))
    scores = mean_scores(rows)
    print(
        f"faithfulness={scores.faithfulness:.3f} "
        f"answer_relevancy={scores.answer_relevancy:.3f} "
        f"context_precision={scores.context_precision:.3f} "
        f"context_recall={scores.context_recall:.3f} "
        f"hit@{RETRIEVAL_K}={hit_at_k(ranks, RETRIEVAL_K):.3f} "
        f"mrr={mean_reciprocal_rank(ranks):.3f}"
    )
    notify_low_faithfulness(scores.faithfulness, settings.alert_webhook_url, _post_webhook)
    if not passes_faithfulness(scores.faithfulness):
        raise SystemExit(
            f"RAGAS faithfulness {scores.faithfulness:.3f} is under {FAITHFULNESS_MINIMUM:.2f}"
        )


def _post_webhook(url: str, payload: dict[str, str]) -> None:
    httpx.post(url, json=payload, timeout=10.0)


if __name__ == "__main__":
    main()
