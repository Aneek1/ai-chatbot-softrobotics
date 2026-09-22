"""Index a JSONL file of documents (one JSON object per line).

Usage: PYTHONUTF8=1 uv run python -m ingest.load_jsonl path/to/documents.jsonl

--index writes somewhere other than the configured index. Use it for anything that is not corpus
data (the test fixtures, a trial run): documents indexed next to the knowledge base cannot be
told apart from it by retrieval, and would answer and be cited in real questions.
"""

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from backend.pipeline.chunk import TokenCounter, chunk_text, load_token_counter
from backend.pipeline.normalize import normalize, script_profile
from backend.pipeline.retrieve import Chunk, ChunkIndex

REQUIRED = ("id", "text", "language", "source", "title", "url", "licence")
OPTIONAL = ("script", "retrieved_at", "revision")


def load_documents(
    lines: Iterable[str],
    index: ChunkIndex,
    count: TokenCounter,
    max_tokens: int = 300,
    overlap_tokens: int = 50,
) -> int:
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError as error:
            # json counts lines within this one string, so its own position
            # is always "line 1"; report the file's line number instead.
            raise ValueError(
                f"line {line_number}: invalid JSON: {error.msg} (column {error.colno})"
            ) from error
        # A line can be valid JSON without being a document ("42", "null").
        if not isinstance(doc, dict):
            raise ValueError(f"line {line_number}: expected a JSON object, got {type(doc).__name__}")
        for field in REQUIRED:
            if field not in doc:
                raise ValueError(f"line {line_number}: missing '{field}'")
            if not isinstance(doc[field], str):
                raise ValueError(f"line {line_number}: '{field}' must be a string")
        for field in OPTIONAL:
            if field in doc and not isinstance(doc[field], str):
                raise ValueError(f"line {line_number}: '{field}' must be a string")
        # Chunk ids derive from the document id, so a repeated id would
        # silently overwrite the earlier document's chunks in the index.
        if doc["id"] in seen:
            raise ValueError(f"line {line_number}: duplicate id '{doc['id']}'")
        seen.add(doc["id"])
        text = normalize(doc["text"])
        script = doc.get("script") or script_profile(text).dominant or ""
        pieces = chunk_text(text, script or None, count, max_tokens, overlap_tokens)
        chunks.extend(
            Chunk(
                id=f"{doc['id']}:{n}",
                doc_id=doc["id"],
                text=piece,
                language=doc["language"],
                source=doc["source"],
                title=doc["title"],
                url=doc["url"],
                licence=doc["licence"],
                script=script,
                retrieved_at=doc.get("retrieved_at", ""),
                revision=doc.get("revision", ""),
            )
            for n, piece in enumerate(pieces)
        )
    index.add(chunks)
    return len(chunks)


def main() -> None:
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from backend.app.config import Settings
    from backend.pipeline.embeddings import E5Embedder

    settings = Settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--index",
        type=Path,
        default=settings.index_dir,
        help="index directory (use a throwaway one for anything that is not corpus data)",
    )
    args = parser.parse_args()

    e5_dir = settings.models_dir / settings.e5_dir
    index = ChunkIndex.open(args.index, E5Embedder(e5_dir))
    try:
        added = load_documents(
            args.path.read_text(encoding="utf-8").splitlines(),
            index,
            count=load_token_counter(e5_dir / "tokenizer.json"),
        )
    finally:
        index.close()
    print(f"Indexed {added} chunks from {args.path} into {args.index}")


if __name__ == "__main__":
    main()
