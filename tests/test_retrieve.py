from qdrant_client import QdrantClient

from backend.pipeline.retrieve import Chunk, ChunkIndex
from tests.fakes import FakeEmbedder


def chunk(n, text, language="eng_Latn"):
    return Chunk(
        id=f"doc{n}:0",
        doc_id=f"doc{n}",
        text=text,
        language=language,
        source="fixture",
        title=f"Doc {n}",
        url=f"https://example.invalid/{n}",
        licence="test-only",
    )


def make_index():
    return ChunkIndex(QdrantClient(location=":memory:"), FakeEmbedder())


def test_search_returns_the_closest_chunk_first():
    index = make_index()
    index.add([chunk(1, "silicone silicone moulding"), chunk(2, "printing TPU"), chunk(3, "market report")])
    hits = index.search("silicone", top_k=2)
    assert hits[0].chunk.doc_id == "doc1"
    assert len(hits) == 2


def test_adding_the_same_chunk_twice_does_not_duplicate():
    index = make_index()
    index.add([chunk(1, "silicone")])
    index.add([chunk(1, "silicone")])
    assert index.count_by_language() == {"eng_Latn": 1}


def test_counts_by_language():
    index = make_index()
    index.add([chunk(1, "silicone"), chunk(2, "silikon", "ind_Latn"), chunk(3, "printing", "ind_Latn")])
    assert index.count_by_language() == {"eng_Latn": 1, "ind_Latn": 2}


def test_empty_index_returns_no_hits():
    assert make_index().search("silicone", top_k=6) == []
