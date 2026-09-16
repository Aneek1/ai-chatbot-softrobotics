"""Build data/knowledge-base.jsonl from the curated CSV, arXiv and Wikipedia.

Usage: PYTHONUTF8=1 uv run python -m ingest.build_kb

This script reaches the network directly. It runs outside the backend process, so the egress guard
does not apply to it: it connects to export.arxiv.org and the *.wikipedia.org API hosts.
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ingest.arxiv import ArxivClient
from ingest.dedup import dedupe
from ingest.documents import Document, today, write_jsonl
from ingest.methods_csv import load_methods
from ingest.wikipedia import WikipediaClient

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHODS = REPO_ROOT / "data" / "methods.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "knowledge-base.jsonl"

QUERIES = (
    'all:"soft robotics" AND all:fabrication',
    'all:"soft actuator" AND all:silicone',
    'all:"soft robot" AND all:"3d printing"',
    'all:"soft gripper" AND all:elastomer',
)
ARTICLES = (
    "Soft robotics",
    "Silicone rubber",
    "Polydimethylsiloxane",
    "Elastomer",
    "Pneumatic actuator",
    "3D printing",
)


class PaperSource(Protocol):
    def search(self, query: str, max_results: int, retrieved_at: str | None = None) -> list[Document]: ...


class ArticleSource(Protocol):
    def articles(self, titles: Sequence[str], retrieved_at: str | None = None) -> list[Document]: ...


@dataclass
class BuildResult:
    documents: list[Document] = field(default_factory=list)
    duplicates: list[tuple[str, str]] = field(default_factory=list)
    unverified_rows: int = 0
    errors: list[str] = field(default_factory=list)


def build(
    *,
    methods_path: Path,
    arxiv_client: PaperSource,
    wikipedia_client: ArticleSource,
    queries: Sequence[str] = QUERIES,
    articles: Sequence[str] = ARTICLES,
    per_query: int = 25,
    retrieved_at: str | None = None,
) -> BuildResult:
    """Collect every source, then deduplicate. A source that fails is recorded, not fatal."""
    when = retrieved_at or today()
    result = BuildResult()
    collected: list[Document] = []

    try:
        documents, unverified = load_methods(methods_path, when)
    except Exception as exc:
        result.errors.append(f"csv: {type(exc).__name__}: {exc}")
    else:
        collected.extend(documents)
        result.unverified_rows = len(unverified)

    for query in queries:
        try:
            collected.extend(arxiv_client.search(query, per_query, when))
        except Exception as exc:
            result.errors.append(f"arxiv ({query}): {type(exc).__name__}: {exc}")

    if articles:
        try:
            collected.extend(wikipedia_client.articles(articles, when))
        except Exception as exc:
            result.errors.append(f"wikipedia: {type(exc).__name__}: {exc}")

    deduplicated = dedupe(collected)
    result.documents = deduplicated.kept
    result.duplicates = deduplicated.dropped
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the knowledge-base JSONL")
    parser.add_argument("--methods", type=Path, default=DEFAULT_METHODS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-query", type=int, default=25)
    args = parser.parse_args()

    result = build(
        methods_path=args.methods,
        arxiv_client=ArxivClient(),
        wikipedia_client=WikipediaClient(),
        per_query=args.per_query,
    )
    written = write_jsonl(args.out, result.documents)
    print(f"Wrote {written} documents to {args.out}")
    print(f"Duplicates dropped: {len(result.duplicates)}")
    print(f"Curated rows not checked against a source, so not indexed: {result.unverified_rows}")
    for error in result.errors:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
