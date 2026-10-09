"""Checks applied to every answer before it leaves the service."""

from egylaw_rag.guardrails.pii import redact_pii

__all__ = ["redact_pii"]
