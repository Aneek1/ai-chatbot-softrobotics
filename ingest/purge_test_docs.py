"""Remove test-only documents from an index.

Usage: PYTHONUTF8=1 uv run python -m ingest.purge_test_docs [--index data/index]

A smoke test that indexed tests/fixtures/documents.jsonl into the real index leaves three
documents behind whose text says it exists only for tests. Retrieval cannot tell them from
corpus data, so they turn up in answers and in evaluation results. This removes them; the
documents themselves stay in the JSONL files they came from.
"""

import argparse
from pathlib import Path

from backend.pipeline.retrieve import ChunkIndex


def purge(index: ChunkIndex) -> tuple[list[str], int]:
    """Returns the document ids removed, sorted, and how many chunks went with them."""
    doc_ids = sorted(index.non_corpus_doc_ids())
    return doc_ids, index.remove_docs(doc_ids)


def main() -> None:
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from backend.app.config import Settings
    from backend.pipeline.embeddings import E5Embedder

    settings = Settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=settings.index_dir, help="index directory")
    args = parser.parse_args()

    index = ChunkIndex.open(args.index, E5Embedder(settings.models_dir / settings.e5_dir))
    try:
        doc_ids, chunks = purge(index)
    finally:
        index.close()
    print(f"Removed {chunks} chunks from {len(doc_ids)} test-only documents: {', '.join(doc_ids) or 'none'}")


if __name__ == "__main__":
    main()
