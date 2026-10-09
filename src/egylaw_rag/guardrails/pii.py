"""Redact Egyptian national IDs and mobile numbers from an answer."""

from __future__ import annotations

import re
from collections.abc import Iterator

# A national ID is 14 digits. A mobile is 010, 011, 012, or 015 plus eight digits.
_NATIONAL_ID = re.compile(r"(?<!\d)\d{14}(?!\d)")
_PHONE = re.compile(r"(?<!\d)(?:\+20|0020|0)1[0125]\d{8}(?!\d)")
_DIGIT_TAIL = re.compile(r"(?:\+20|0020|0)?\d{0,16}$")


def redact_pii(text: str) -> str:
    cleaned = _NATIONAL_ID.sub("[REDACTED]", text)
    return _PHONE.sub("[REDACTED]", cleaned)


def iter_redacted(pieces: Iterator[str]) -> Iterator[str]:
    """Redact streamed pieces, holding a trailing digit run that may still grow."""
    pending = ""
    for piece in pieces:
        text = pending + piece
        tail = _DIGIT_TAIL.search(text)
        hold = tail.group() if tail and any(char.isdigit() for char in tail.group()) else ""
        if hold and len(re.sub(r"\D", "", hold)) < 14 and text.endswith(hold):
            head = text[: -len(hold)] if hold else text
            pending = hold
        else:
            head = text
            pending = ""
        cleaned = redact_pii(head)
        if cleaned:
            yield cleaned
    if pending:
        yield redact_pii(pending)
