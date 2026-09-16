"""Exact deduplication by hash of the normalized text (spec 6.3)."""

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field

from backend.pipeline.normalize import normalize
from ingest.documents import Document


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DedupeResult:
    kept: list[Document] = field(default_factory=list)
    dropped: list[tuple[str, str]] = field(default_factory=list)  # (dropped id, id kept instead)


def dedupe(documents: Iterable[Document]) -> DedupeResult:
    """Keep the first document with a given id and text; report every later copy."""
    result = DedupeResult()
    first_with: dict[str, str] = {}
    seen_ids: set[str] = set()
    for document in documents:
        if document.id in seen_ids:
            result.dropped.append((document.id, document.id))
            continue
        digest = text_hash(document.text)
        if digest in first_with:
            result.dropped.append((document.id, first_with[digest]))
            continue
        seen_ids.add(document.id)
        first_with[digest] = document.id
        result.kept.append(document)
    return result
