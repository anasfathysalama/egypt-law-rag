from fastapi.testclient import TestClient

from egylaw_rag.api.app import create_app
from egylaw_rag.api.generate import MissingApiKey
from egylaw_rag.index.chunk import ArticleChunk, chunk_articles
from egylaw_rag.index.store import build_index
from egylaw_rag.observe.trace import MemoryTracer
from tests.test_index import _article, _KeywordEmbedder


class _FixedGenerator:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def generate(self, question: str, chunks: list[ArticleChunk]) -> str:
        self.questions.append(question)
        return f"Answer from article {chunks[0].article_number}."


class _NeedsKey:
    def generate(self, question: str, chunks: list[ArticleChunk]) -> str:
        raise MissingApiKey("Set GEMINI_API_KEY in .env before calling /ask.")


def _client(generator: object) -> TestClient:
    articles = [
        _article(147, "العقد شريعة المتعاقدين contract"),
        _article(54, "repealed by decree", repealed=True),
    ]
    index = build_index(chunk_articles(articles), _KeywordEmbedder())
    app = create_app(index=index, embedder=_KeywordEmbedder(), generator=generator)  # type: ignore[arg-type]
    return TestClient(app)


def test_health_counts_indexed_chunks() -> None:
    response = _client(_FixedGenerator()).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "documents_indexed": 2}


def test_empty_question_is_rejected() -> None:
    client = _client(_FixedGenerator())
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "   "}).status_code == 422


def test_named_article_returns_citation() -> None:
    generator = _FixedGenerator()
    response = _client(generator).post("/ask", json={"question": "What does Article 147 say?"})
    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == ["Egyptian Civil Code, Article 147"]
    assert "147" in body["answer"]
    assert generator.questions == ["What does Article 147 say?"]


def test_repealed_article_does_not_call_the_model() -> None:
    generator = _FixedGenerator()
    response = _client(generator).post("/ask", json={"question": "المادة ٥٤"})
    assert response.status_code == 200
    body = response.json()
    assert "ملغاة" in body["answer"]
    assert body["sources"] == ["Egyptian Civil Code, Article 54"]
    assert generator.questions == []


def test_stream_yields_tokens_in_order() -> None:
    class _Streamer(_FixedGenerator):
        def stream(self, question: str, chunks: list[ArticleChunk]):
            yield "Hello "
            yield "147"

    client = _client(_Streamer())
    response = client.post("/ask", json={"question": "contract", "stream": True})
    assert response.status_code == 200
    assert response.text == "Hello 147"


def test_answer_redacts_a_national_id() -> None:
    class _Leaky:
        def generate(self, question: str, chunks: list[ArticleChunk]) -> str:
            return "The id is 29901011234567."

    response = _client(_Leaky()).post("/ask", json={"question": "contract"})
    assert response.status_code == 200
    assert "29901011234567" not in response.json()["answer"]
    assert "[REDACTED]" in response.json()["answer"]


def test_ask_records_a_trace_and_metrics() -> None:
    tracer = MemoryTracer()
    articles = [_article(147, "العقد شريعة المتعاقدين contract")]
    index = build_index(chunk_articles(articles), _KeywordEmbedder())
    app = create_app(
        index=index,
        embedder=_KeywordEmbedder(),
        generator=_FixedGenerator(),
        tracer=tracer,
    )
    client = TestClient(app)
    response = client.post("/ask", json={"question": "contract"})
    assert response.status_code == 200
    assert tracer.spans == ["retrieve", "generate"]
    assert "faithfulness" in tracer.scores
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "egylaw_tokens_total" in metrics.text
    assert "egylaw_query_drift" in metrics.text


def test_missing_gemini_key_is_unavailable() -> None:
    response = _client(_NeedsKey()).post("/ask", json={"question": "contract"})
    assert response.status_code == 503
