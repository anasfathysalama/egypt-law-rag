"""Read and write the corpus files the DVC stages pass between them."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from egylaw_rag.corpus.extract import RawRow
from egylaw_rag.corpus.models import ArticleRecord


def write_rows(path: Path, rows: list[RawRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(row) for row in rows]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def read_rows(path: Path) -> list[RawRow]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [RawRow(**item) for item in payload]


def write_articles(path: Path, articles: list[ArticleRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [article.model_dump() for article in articles]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_articles(path: Path) -> list[ArticleRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [ArticleRecord.model_validate(item) for item in payload]
