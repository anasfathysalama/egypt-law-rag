"""Build the article list from the source PDF."""

from __future__ import annotations

from pathlib import Path

from egylaw_rag.corpus.extract import extract_rows
from egylaw_rag.corpus.models import ArticleRecord
from egylaw_rag.corpus.parse import parse_rows


def build_corpus(pdf_path: Path) -> list[ArticleRecord]:
    return parse_rows(extract_rows(pdf_path))
