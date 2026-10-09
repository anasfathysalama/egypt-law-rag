"""DVC stage: PDF table rows, saved so parsing does not re-read the PDF."""

from __future__ import annotations

from egylaw_rag.config import get_settings
from egylaw_rag.corpus.extract import extract_rows
from egylaw_rag.corpus.store import write_rows

PDF_PATH = get_settings().pdf
ROWS_PATH = get_settings().raw_rows


def main() -> None:
    rows = extract_rows(PDF_PATH)
    write_rows(ROWS_PATH, rows)
    print(f"wrote {len(rows)} rows to {ROWS_PATH}")


if __name__ == "__main__":
    main()
