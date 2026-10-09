"""Score the CI questions and stop the build when faithfulness is under 0.75."""

from __future__ import annotations

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
from egylaw_rag.index.embed import LocalEmbedder
from egylaw_rag.index.store import VectorIndex
from egylaw_rag.tracking import faithfulness_score


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
    scores: list[float] = []
    ranks: list[int | None] = []
    for question in questions:
        article = articles[question.article_number]
        hits = index.search(embedder.embed_query(question.question_ar), k=RETRIEVAL_K)
        scores.append(faithfulness_score(article.text_normalized, [hit.text for hit in hits]))
        numbers = [hit.article_number for hit in hits]
        ranks.append(first_relevant_rank(numbers, question.article_number))
    faithfulness = sum(scores) / len(scores)
    print(
        f"faithfulness={faithfulness:.3f} "
        f"hit@{RETRIEVAL_K}={hit_at_k(ranks, RETRIEVAL_K):.3f} "
        f"mrr={mean_reciprocal_rank(ranks):.3f}"
    )
    if not passes_faithfulness(faithfulness):
        raise SystemExit(
            f"faithfulness {faithfulness:.3f} is under {FAITHFULNESS_MINIMUM:.2f}"
        )


if __name__ == "__main__":
    main()
