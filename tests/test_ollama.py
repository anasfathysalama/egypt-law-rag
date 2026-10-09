from egylaw_rag.api.ollama import OllamaGenerator, _delta_text
from egylaw_rag.index.chunk import ArticleChunk


def _chunk() -> ArticleChunk:
    return ArticleChunk(
        chunk_id="147",
        article_number=147,
        citation="Egyptian Civil Code, Article 147",
        text="العقد شريعة المتعاقدين",
        is_repealed=False,
    )


def test_ollama_generate_reads_the_message() -> None:
    def post(url: str, payload: dict[str, object], headers: dict[str, str]) -> dict[str, object]:
        assert url.endswith("/chat/completions")
        assert payload["model"] == "qwen2.5:0.5b"
        assert payload["stream"] is False
        return {"choices": [{"message": {"content": "The contract binds."}}]}

    generator = OllamaGenerator(model="qwen2.5:0.5b", post=post)
    assert generator.generate("contract", [_chunk()]) == "The contract binds."


def test_ollama_stream_yields_tokens() -> None:
    lines = iter(
        [
            'data: {"choices":[{"delta":{"content":"The "}}]}',
            'data: {"choices":[{"delta":{"content":"contract"}}]}',
            "data: [DONE]",
        ]
    )
    generator = OllamaGenerator(model="qwen2.5:0.5b", stream_lines=lambda url, payload: lines)
    assert list(generator.stream("contract", [_chunk()])) == ["The ", "contract"]
    assert _delta_text("data: [DONE]") == ""
