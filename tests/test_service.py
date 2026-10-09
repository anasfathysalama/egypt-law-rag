from egylaw_rag.api.service import answer_question
from egylaw_rag.index.chunk import ArticleChunk, chunk_articles
from egylaw_rag.index.store import build_index
from tests.test_index import _article, _KeywordEmbedder


class _FixedGenerator:
    def generate(self, question: str, chunks: list[ArticleChunk]) -> str:
        return "from the model"


def test_open_question_returns_the_model_answer_and_citation() -> None:
    index = build_index(
        chunk_articles([_article(147, "contract")]),
        _KeywordEmbedder(),
    )
    answer, sources = answer_question(
        index,
        _KeywordEmbedder(),
        _FixedGenerator(),
        "contract",
        k=1,
    )
    assert answer == "from the model"
    assert sources == ["Egyptian Civil Code, Article 147"]
