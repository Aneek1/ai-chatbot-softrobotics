import uuid
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams

from backend.pipeline.embeddings import Embedder


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    text: str
    language: str
    source: str
    title: str
    url: str
    licence: str
    # Provenance (spec 6.1). Empty for chunks indexed before these fields existed.
    script: str = ""
    retrieved_at: str = ""
    revision: str = ""


# The test fixtures in tests/fixtures/documents.jsonl carry these markers. Indexing them into
# the real index (a smoke test run without an index path of its own) would let a sentence that
# "exists only for tests" be retrieved for, and cited in, an answer to a real question.
TEST_ONLY_SOURCES = frozenset({"fixture"})
TEST_ONLY_LICENCES = frozenset({"test-only"})


def is_test_only(source: str, licence: str) -> bool:
    """True for a chunk that is test data rather than corpus data."""
    return source in TEST_ONLY_SOURCES or licence in TEST_ONLY_LICENCES


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> list[Hit]: ...


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class ChunkIndex:
    COLLECTION = "chunks"

    def __init__(self, client: QdrantClient, embedder: Embedder):
        self._client = client
        self._embedder = embedder
        if not client.collection_exists(self.COLLECTION):
            client.create_collection(
                self.COLLECTION,
                vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE),
            )

    @classmethod
    def open(cls, path: Path, embedder: Embedder) -> "ChunkIndex":
        path.mkdir(parents=True, exist_ok=True)
        return cls(QdrantClient(path=str(path)), embedder)

    def add(self, chunks: Sequence[Chunk], batch_size: int = 64) -> None:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self._embedder.embed_passages([c.text for c in batch])
            points = [
                PointStruct(id=_point_id(c.id), vector=v, payload=asdict(c))
                for c, v in zip(batch, vectors, strict=True)
            ]
            self._client.upsert(self.COLLECTION, points=points)

    def search(self, query: str, top_k: int) -> list[Hit]:
        vector = self._embedder.embed_queries([query])[0]
        response = self._client.query_points(self.COLLECTION, query=vector, limit=top_k, with_payload=True)
        return [Hit(chunk=Chunk(**p.payload), score=p.score) for p in response.points]

    def count_by_language(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        offset = None
        while True:
            points, offset = self._client.scroll(
                self.COLLECTION, limit=256, offset=offset, with_payload=["language"], with_vectors=False
            )
            counts.update(p.payload["language"] for p in points)
            if offset is None:
                return dict(counts)

    def doc_ids(self) -> set[str]:
        ids: set[str] = set()
        offset = None
        while True:
            points, offset = self._client.scroll(
                self.COLLECTION, limit=256, offset=offset, with_payload=["doc_id"], with_vectors=False
            )
            ids.update(p.payload["doc_id"] for p in points)
            if offset is None:
                return ids

    def non_corpus_doc_ids(self) -> set[str]:
        """Documents in the index that are test data (spec 6.1: the index holds corpus only)."""
        ids: set[str] = set()
        offset = None
        while True:
            points, offset = self._client.scroll(
                self.COLLECTION,
                limit=256,
                offset=offset,
                with_payload=["doc_id", "source", "licence"],
                with_vectors=False,
            )
            ids.update(
                p.payload["doc_id"]
                for p in points
                if is_test_only(p.payload.get("source", ""), p.payload.get("licence", ""))
            )
            if offset is None:
                return ids

    def remove_docs(self, doc_ids: Iterable[str]) -> int:
        """Delete every chunk of the named documents; returns how many chunks went."""
        wanted = set(doc_ids)
        if not wanted:
            return 0
        doomed: list = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                self.COLLECTION, limit=256, offset=offset, with_payload=["doc_id"], with_vectors=False
            )
            doomed += [p.id for p in points if p.payload["doc_id"] in wanted]
            if offset is None:
                break
        if doomed:
            self._client.delete(self.COLLECTION, points_selector=PointIdsList(points=doomed))
        return len(doomed)

    def close(self) -> None:
        self._client.close()
