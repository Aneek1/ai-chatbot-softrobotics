import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

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

    def close(self) -> None:
        self._client.close()
