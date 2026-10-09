from egylaw_rag.observe.trace import LangfuseTracer, MemoryTracer


def test_memory_tracer_records_spans_and_faithfulness() -> None:
    tracer = MemoryTracer()
    tracer.span("retrieve")
    tracer.span("generate")
    tracer.score("faithfulness", 0.91)
    assert tracer.spans == ["retrieve", "generate"]
    assert tracer.scores["faithfulness"] == 0.91


def test_langfuse_tracer_posts_the_faithfulness_score() -> None:
    sent: list[dict[str, object]] = []

    def post(host: str, payload: dict[str, object]) -> None:
        sent.append({"host": host, **payload})

    tracer = LangfuseTracer("http://langfuse:3000", "pk", "sk", post=post)
    tracer.span("retrieve")
    tracer.score("faithfulness", 0.88)
    assert sent[0]["name"] == "retrieve"
    assert sent[1]["value"] == 0.88
    assert sent[1]["host"] == "http://langfuse:3000"
