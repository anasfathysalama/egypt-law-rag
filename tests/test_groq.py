import pytest

from egylaw_rag.api.generate import MissingApiKey
from egylaw_rag.api.groq import GroqGenerator
from egylaw_rag.index.chunk import chunk_articles
from tests.test_index import _article


def test_groq_requires_an_api_key() -> None:
    chunk = chunk_articles([_article(147, "contract")])[0]
    generator = GroqGenerator(api_key="")
    with pytest.raises(MissingApiKey):
        generator.generate("Article 147", [chunk])


def test_groq_reads_the_qwen_answer() -> None:
    chunk = chunk_articles([_article(147, "The contract is the law of the parties.")])[0]

    def post(url: str, payload: dict[str, object], headers: dict[str, str]) -> dict[str, object]:
        assert url.endswith("/chat/completions")
        assert payload["model"] == "qwen/qwen3-32b"
        assert payload["reasoning_effort"] == "none"
        assert headers["Authorization"] == "Bearer test-key"
        messages = payload["messages"]
        assert isinstance(messages, list)
        assert "Egyptian Civil Code, Article 147" in messages[0]["content"]
        return {"choices": [{"message": {"content": "The parties are bound by the contract."}}]}

    generator = GroqGenerator(api_key="test-key", model="qwen/qwen3-32b", post=post)
    assert generator.generate("Article 147", [chunk]) == "The parties are bound by the contract."
