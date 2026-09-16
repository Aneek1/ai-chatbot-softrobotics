"""Read the curated fabrication-methods CSV.

A row becomes a document only when a person has checked it against a source that anyone can open.
Rows marked unverified are reported and never indexed.

Usage: PYTHONUTF8=1 uv run python -m ingest.methods_csv data/methods.csv
"""

import argparse
import csv
import re
from collections.abc import Sequence
from pathlib import Path

from ingest.documents import Document, IngestError, make_document, today

COLUMNS = (
    "method",
    "description",
    "materials",
    "process",
    "language",
    "source_title",
    "source_url",
    "licence",
    "status",
    "checked_by",
    "checked_on",
    "notes",
)
STATUSES = ("verified", "unverified")
VERIFIED_NEEDS = (
    "method",
    "description",
    "language",
    "source_title",
    "source_url",
    "licence",
    "checked_by",
    "checked_on",
)
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NOT_SLUG = re.compile(r"[^a-z0-9]+")

Row = dict[str, str]


def read_rows(text: str) -> list[Row]:
    reader = csv.DictReader(text.splitlines())
    missing = sorted(set(COLUMNS) - set(reader.fieldnames or []))
    if missing:
        raise IngestError(f"methods CSV is missing columns: {', '.join(missing)}")
    return [{name: (row.get(name) or "").strip() for name in COLUMNS} for row in reader]


def slug(method: str) -> str:
    return _NOT_SLUG.sub("-", method.lower()).strip("-")


def document_text(row: Row) -> str:
    parts = [f"{row['method']}. {row['description']}"]
    if row["materials"]:
        parts.append(f"Materials: {row['materials']}.")
    if row["process"]:
        parts.append(f"Process: {row['process']}.")
    parts.append(f"Source: {row['source_title']} ({row['source_url']}).")
    return " ".join(parts)


def build_documents(rows: Sequence[Row], retrieved_at: str) -> tuple[list[Document], list[Row]]:
    documents: list[Document] = []
    unverified: list[Row] = []
    for number, row in enumerate(rows, start=2):  # line 1 is the header
        where = f"row {number} ({row['method'] or 'no method'})"
        if row["status"] not in STATUSES:
            raise IngestError(f"{where}: status must be one of {', '.join(STATUSES)}, got '{row['status']}'")
        if row["status"] == "unverified":
            unverified.append(row)
            continue
        for name in VERIFIED_NEEDS:
            if not row[name]:
                raise IngestError(f"{where}: verified rows need '{name}'")
        if not row["source_url"].startswith(("http://", "https://")):
            raise IngestError(f"{where}: source_url must be an http(s) link, got '{row['source_url']}'")
        if not _DATE.match(row["checked_on"]):
            raise IngestError(f"{where}: checked_on must be YYYY-MM-DD, got '{row['checked_on']}'")
        documents.append(
            make_document(
                id=f"csv:{slug(row['method'])}",
                text=document_text(row),
                language=row["language"],
                source="csv",
                title=row["method"],
                url=row["source_url"],
                licence=row["licence"],
                retrieved_at=retrieved_at,
                revision=f"checked {row['checked_on']} by {row['checked_by']}",
            )
        )
    return documents, unverified


def load_methods(path: Path, retrieved_at: str | None = None) -> tuple[list[Document], list[Row]]:
    rows = read_rows(path.read_text(encoding="utf-8"))
    return build_documents(rows, retrieved_at or today())


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the curated methods CSV")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    documents, unverified = load_methods(args.path)
    total = len(documents) + len(unverified)
    print(f"{total} rows: {len(documents)} verified, {len(unverified)} unverified")
    for row in unverified:
        print(f"  unverified, not indexed: {row['method']}")


if __name__ == "__main__":
    main()
