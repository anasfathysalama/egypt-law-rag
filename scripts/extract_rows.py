"""DVC stage: PDF table rows, saved so parsing does not re-read the PDF."""

from __future__ import annotations

from pathlib import Path

from egylaw_rag.corpus.extract import extract_rows
from egylaw_rag.corpus.store import write_rows

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "data" / "raw" / "law.pdf"
ROWS_PATH = ROOT / "data" / "interim" / "raw_rows.json"


def main() -> None:
    rows = extract_rows(PDF_PATH)
    write_rows(ROWS_PATH, rows)
    print(f"wrote {len(rows)} rows to {ROWS_PATH}")


if __name__ == "__main__":
    main()
