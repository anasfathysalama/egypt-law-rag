import pytest

from egylaw_rag.api.generate import GeminiGenerator, MissingApiKey, build_prompt
from egylaw_rag.index.chunk import chunk_articles
from tests.test_index import _article


def test_prompt_includes_the_citation_and_question() -> None:
    chunk = chunk_articles([_article(147, "العقد شريعة المتعاقدين")])[0]
    prompt = build_prompt("What binds the parties?", [chunk])
    assert "Egyptian Civil Code, Article 147" in prompt
    assert "What binds the parties?" in prompt
    assert "in force" in prompt
    assert "Write the entire answer in English." in prompt


def test_arabic_question_asks_for_an_arabic_answer() -> None:
    chunk = chunk_articles([_article(147, "العقد شريعة المتعاقدين")])[0]
    prompt = build_prompt("ماذا تقول المادة ١٤٧؟", [chunk])
    assert "Write the entire answer in Arabic." in prompt


def test_gemini_requires_an_api_key() -> None:
    chunk = chunk_articles([_article(1, "text")])[0]
    generator = GeminiGenerator(api_key="")
    with pytest.raises(MissingApiKey):
        generator.generate("Article 1", [chunk])


def test_gemini_reads_the_answer_text() -> None:
    chunk = chunk_articles([_article(147, "contract")])[0]

    def post(url: str, payload: dict[str, object], headers: dict[str, str]) -> dict[str, object]:
        assert url.endswith("/models/gemini-test:generateContent")
        assert headers["x-goog-api-key"] == "test-key"
        assert isinstance(payload["contents"], list)
        return {"candidates": [{"content": {"parts": [{"text": "The contract binds them."}]}}]}

    generator = GeminiGenerator(api_key="test-key", model="gemini-test", post=post)
    assert generator.generate("Article 147", [chunk]) == "The contract binds them."
