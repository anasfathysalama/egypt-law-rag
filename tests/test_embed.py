from egylaw_rag.index.embed import EMBEDDING_MODELS, SECOND_EMBEDDING_MODEL, LocalEmbedder


def test_two_cpu_embedding_models_are_available() -> None:
    assert len(EMBEDDING_MODELS) == 2
    embedder = LocalEmbedder(SECOND_EMBEDDING_MODEL)
    assert embedder.model_name == SECOND_EMBEDDING_MODEL
    assert embedder._model is None
