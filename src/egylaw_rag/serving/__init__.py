"""Serving entry points. FastAPI is the Compose app. BentoML exposes the same core."""

from egylaw_rag.serving.ask import ask_async

__all__ = ["ask_async"]
