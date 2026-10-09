from egylaw_rag.config import Settings
from egylaw_rag.corpus.store import read_articles
from egylaw_rag.eval.metrics import (
    FAITHFULNESS_MINIMUM,
    RETRIEVAL_K,
    EvalQuestion,
    first_relevant_rank,
    hit_at_k,
    load_questions,
    mean_reciprocal_rank,
    passes_faithfulness,
)
from egylaw_rag.index.chunk import chunk_articles
from egylaw_rag.index.store import build_index, search_text
from tests.test_index import _article, _KeywordEmbedder


def test_hit_at_k_and_mrr_ignore_misses() -> None:
    assert first_relevant_rank([10, 147, 54], 147) == 2
    assert first_relevant_rank([10], 147) is None
    assert hit_at_k([1, 2, None], k=1) == 1 / 3
    assert hit_at_k([1, 2, None], k=2) == 2 / 3
    assert mean_reciprocal_rank([1, 2, None]) == (1 + 0.5) / 3
    assert hit_at_k([], k=5) == 0.0
    assert mean_reciprocal_rank([]) == 0.0


def test_retrieval_on_a_golden_set_needs_no_language_model() -> None:
    articles = [
        _article(147, "contract"),
        _article(54, "repealed", repealed=True),
    ]
    index = build_index(chunk_articles(articles), _KeywordEmbedder())
    questions = [
        EvalQuestion(
            id="hit",
            question_ar="contract",
            question_en="contract",
            article_number=147,
        ),
        EvalQuestion(
            id="miss",
            question_ar="contract",
            question_en="contract",
            article_number=54,
        ),
    ]
    ranks = []
    for question in questions:
        hits = search_text(index, _KeywordEmbedder(), question.question_en, k=1)
        numbers = [hit.article_number for hit in hits]
        ranks.append(first_relevant_rank(numbers, question.article_number))
    assert ranks == [1, None]
    assert hit_at_k(ranks, k=1) == 0.5
    assert mean_reciprocal_rank(ranks) == 0.5


def test_faithfulness_gate_is_0_75() -> None:
    assert FAITHFULNESS_MINIMUM == 0.75
    assert RETRIEVAL_K == 16
    assert passes_faithfulness(0.75)
    assert not passes_faithfulness(0.749)


def test_question_files_cover_ci_and_the_full_run() -> None:
    settings = Settings(_env_file=None)
    ci = load_questions(settings.questions_ci)
    full = load_questions(settings.questions_full)
    fixture_numbers = {
        article.article_number for article in read_articles(settings.fixture)
    }
    assert len(ci) == 20
    assert len(full) == 50
    assert {question.id for question in full} == {f"q{number:02d}" for number in range(1, 51)}
    assert {question.article_number for question in ci} <= fixture_numbers
    assert {question.article_number for question in ci} <= {
        question.article_number for question in full
    }
    assert all(question.question_ar and question.question_en for question in full)
