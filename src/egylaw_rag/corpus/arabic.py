"""Repair visual-order Arabic from the Civil Code PDF and normalize it for search."""

from __future__ import annotations

import re
import unicodedata

_DIGITS = re.compile(r"[0-9\u0660-\u0669]+")
_WRAPPED_NUMBER = re.compile(r"[()]\s*([0-9\u0660-\u0669]+)\s*[()]")
_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u0640]")
_ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
_LAM_ALEF = (
    ("اإل", "الإ"),
    ("األ", "الأ"),
    ("اآل", "الآ"),
)
# One-letter clitics attach to the definite article, so "بالعقد" must stay.
_CLITIC = set("وبفكل")
_ARABIC_LETTER = re.compile(r"[\u0621-\u063A\u0641-\u064A]")


def _fix_inner_lam_alef(text: str) -> str:
    """Swap a lam-alef ligature that the extractor stored as alef then lam.

    The definite article ``ال`` is left in place, including after ``و ب ف ك ل``.
    """
    chars = list(text)
    index = 1
    while index < len(chars) - 1:
        previous = chars[index - 1]
        if (
            chars[index] == "ا"
            and chars[index + 1] == "ل"
            and _ARABIC_LETTER.match(previous)
            and previous not in _CLITIC
        ):
            chars[index] = "ل"
            chars[index + 1] = "ا"
            index += 2
            continue
        index += 1
    return "".join(chars)


def repair_line(line: str) -> str:
    """Turn one visually stored Arabic line into logical reading order.

    pdfplumber returns each line reversed, while digit runs stay in logical
    order. Paragraph numbers are wrapped in parentheses that may face either
    way, so any paren pair around a number is rewritten as ``(n)``.
    """
    restored = line[::-1]
    restored = _DIGITS.sub(lambda match: match.group(0)[::-1], restored)
    for broken, fixed in _LAM_ALEF:
        restored = restored.replace(broken, fixed)
    restored = _WRAPPED_NUMBER.sub(r"(\1)", restored)
    return _fix_inner_lam_alef(restored).strip()


def repair_arabic(text: str) -> str:
    """Repair every line and drop lines that are empty after repair."""
    lines = [repair_line(line) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def normalize_arabic(text: str) -> str:
    """Build the embedding form without changing letters that carry meaning.

    Diacritics and alef/hamza-on-alef variants are unified. ``ة`` / ``ه`` and
    ``ى`` / ``ي`` are left as written.
    """
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _DIACRITICS.sub("", normalized)
    normalized = normalized.translate(_ALEF)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized
