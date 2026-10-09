"""Civil Code corpus: PDF rows to one record per article."""

from egylaw_rag.corpus.build import build_corpus
from egylaw_rag.corpus.models import ArticleRecord

__all__ = ["ArticleRecord", "build_corpus"]
