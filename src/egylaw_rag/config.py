"""Paths and settings. Defaults match dvc.yaml. Override them with a .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_default_root() / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Field(default_factory=_default_root)
    pdf_path: Path = Path("data/raw/law.pdf")
    raw_rows_path: Path = Path("data/interim/raw_rows.json")
    articles_path: Path = Path("data/processed/civil_code.json")
    index_dir: Path = Path("data/index")
    fixture_path: Path = Path("tests/fixtures/articles_sample.json")
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_model_alt: str = "minishlab/potion-multilingual-128M"
    questions_ci_path: Path = Path("data/eval/questions_ci.json")
    questions_full_path: Path = Path("data/eval/questions_full.json")
    groq_api_key: str = ""
    groq_model: str = "qwen3.8-27b"
    groq_api_base: str = "https://api.groq.com/openai/v1"
    groq_max_tokens: int = 512
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    embedding_cache: Path = Path(".cache/fastembed")
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    mlflow_experiment: str = "civil-code-retrieval"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    gemini_api_base: str = "https://generativelanguage.googleapis.com/v1beta"

    def resolve(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return self.project_root / path

    @property
    def pdf(self) -> Path:
        return self.resolve(self.pdf_path)

    @property
    def raw_rows(self) -> Path:
        return self.resolve(self.raw_rows_path)

    @property
    def articles(self) -> Path:
        return self.resolve(self.articles_path)

    @property
    def index(self) -> Path:
        return self.resolve(self.index_dir)

    @property
    def fixture(self) -> Path:
        return self.resolve(self.fixture_path)

    @property
    def embedding_cache_dir(self) -> Path:
        return self.resolve(self.embedding_cache)

    @property
    def questions_ci(self) -> Path:
        return self.resolve(self.questions_ci_path)

    @property
    def questions_full(self) -> Path:
        return self.resolve(self.questions_full_path)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
