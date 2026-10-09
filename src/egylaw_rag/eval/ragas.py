"""The four RAGAS scores, plus the alert when faithfulness drops under 0.80.

Faithfulness is the share of answer statements that the retrieved articles support.
Answer relevancy is the share of question words that the answer uses.
Context precision is the average precision of the gold article in the ranked hits.
Context recall is the share of the gold article's words found in the retrieved hits.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

_TOKEN = re.compile(r"[\w\u0600-\u06FF]{3,}", re.UNICODE)
_SENTENCE = re.compile(r"[^.!?؟\n]+")
ALERT_BELOW = 0.80
_SUPPORT = 0.5


@dataclass(frozen=True)
class RagasScores:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def _statements(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE.findall(text) if part.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def ragas_faithfulness(answer: str, contexts: list[str]) -> float:
    """Share of answer statements supported by the retrieved context."""
    context_tokens = _tokens(" ".join(contexts))
    statements = _statements(answer)
    if not statements:
        return 0.0
    supported = 0
    for statement in statements:
        words = _tokens(statement)
        if not words:
            continue
        if len(words & context_tokens) / len(words) >= _SUPPORT:
            supported += 1
    return supported / len(statements)


def answer_relevancy(question: str, answer: str) -> float:
    """Share of the question's words that also appear in the answer."""
    question_words = _tokens(question)
    if not question_words:
        return 0.0
    return len(question_words & _tokens(answer)) / len(question_words)


def context_precision(retrieved: list[int], gold: int) -> float:
    """Average precision of the gold article inside the ranked hit list."""
    relevant = 0
    precision_sum = 0.0
    for rank, number in enumerate(retrieved, start=1):
        if number != gold:
            continue
        relevant += 1
        precision_sum += relevant / rank
    if relevant == 0:
        return 0.0
    return precision_sum / relevant


def context_recall(reference: str, contexts: list[str]) -> float:
    """Share of the reference article's words found in the retrieved hits."""
    reference_words = _tokens(reference)
    if not reference_words:
        return 0.0
    return len(reference_words & _tokens(" ".join(contexts))) / len(reference_words)


def score_example(
    question: str,
    answer: str,
    contexts: list[str],
    retrieved: list[int],
    gold: int,
    reference: str,
) -> RagasScores:
    return RagasScores(
        faithfulness=ragas_faithfulness(answer, contexts),
        answer_relevancy=answer_relevancy(question, answer),
        context_precision=context_precision(retrieved, gold),
        context_recall=context_recall(reference, contexts),
    )


def mean_scores(rows: list[RagasScores]) -> RagasScores:
    if not rows:
        return RagasScores(0.0, 0.0, 0.0, 0.0)
    count = len(rows)
    return RagasScores(
        faithfulness=sum(row.faithfulness for row in rows) / count,
        answer_relevancy=sum(row.answer_relevancy for row in rows) / count,
        context_precision=sum(row.context_precision for row in rows) / count,
        context_recall=sum(row.context_recall for row in rows) / count,
    )


def notify_low_faithfulness(
    score: float,
    webhook_url: str,
    post: Callable[[str, dict[str, str]], None],
    minimum: float = ALERT_BELOW,
) -> bool:
    """POST a Discord-compatible message when faithfulness is under 0.80."""
    if not webhook_url or score >= minimum:
        return False
    post(
        webhook_url,
        {"content": f"RAGAS faithfulness {score:.3f} is under {minimum:.2f}"},
    )
    return True
