"""Read Civil Code PDF tables into English and Arabic cell text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

_ARABIC = re.compile(r"[\u0600-\u06FF]")
_LATIN = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class RawRow:
    page: int
    english: str
    arabic: str


def _split_languages(cells: list[str | None]) -> tuple[str, str]:
    english_parts: list[str] = []
    arabic_parts: list[str] = []
    for cell in cells:
        text = (cell or "").strip()
        if not text:
            continue
        arabic_letters = len(_ARABIC.findall(text))
        latin_letters = len(_LATIN.findall(text))
        if arabic_letters > latin_letters:
            arabic_parts.append(text)
        else:
            english_parts.append(text)
    return "\n".join(english_parts), "\n".join(arabic_parts)


def extract_rows(pdf_path: Path) -> list[RawRow]:
    """Return every table row. English and Arabic are chosen by script, not column index."""
    rows: list[RawRow] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            for table in page.extract_tables() or []:
                for cells in table:
                    english, arabic = _split_languages(list(cells))
                    if not english and not arabic:
                        continue
                    rows.append(RawRow(page=page_number, english=english, arabic=arabic))
    return rows
