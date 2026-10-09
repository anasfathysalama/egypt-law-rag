"""hit@k and MRR for a golden article set. No language-model judge."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

FAITHFULNESS_MINIMUM = 0.75
RETRIEVAL_K = 8


class EvalQuestion(BaseModel):
    id: str
    question_ar: str
    question_en: str
    article_number: int = Field(ge=1)


def load_questions(path: Path) -> list[EvalQuestion]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [EvalQuestion.model_validate(item) for item in payload]


def first_relevant_rank(article_numbers: list[int], gold: int) -> int | None:
    """1-based rank of the gold article, or None when it is missing."""
    for rank, number in enumerate(article_numbers, start=1):
        if number == gold:
            return rank
    return None


def hit_at_k(ranks: list[int | None], k: int) -> float:
    """Share of questions whose gold article appears at rank k or better."""
    if k < 1:
        raise ValueError("k must be positive")
    if not ranks:
        return 0.0
    hits = sum(1 for rank in ranks if rank is not None and rank <= k)
    return hits / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    """Average of 1/rank. A miss contributes zero."""
    if not ranks:
        return 0.0
    total = sum(1.0 / rank if rank is not None else 0.0 for rank in ranks)
    return total / len(ranks)


def passes_faithfulness(score: float, minimum: float = FAITHFULNESS_MINIMUM) -> bool:
    return score >= minimum
