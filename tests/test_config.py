from egylaw_rag.config import Settings


def test_default_paths_stay_inside_the_project(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("PDF_PATH", raising=False)
    monkeypatch.delenv("ARTICLES_PATH", raising=False)
    monkeypatch.delenv("INDEX_DIR", raising=False)
    monkeypatch.delenv("EMBEDDING_CACHE", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT", raising=False)
    settings = Settings(_env_file=None)
    assert settings.pdf == settings.project_root / "data" / "raw" / "law.pdf"
    assert settings.articles == settings.project_root / "data" / "processed" / "civil_code.json"
    assert settings.index == settings.project_root / "data" / "index"
    assert settings.api_port == 8000
    assert settings.gemini_api_key == ""
    assert settings.groq_api_key == ""
    assert settings.groq_model == "qwen3.8-27b"
    assert settings.embedding_cache_dir == settings.project_root / ".cache" / "fastembed"
    assert settings.mlflow_tracking_uri == "http://127.0.0.1:5000"
    assert settings.mlflow_experiment == "civil-code-retrieval"
