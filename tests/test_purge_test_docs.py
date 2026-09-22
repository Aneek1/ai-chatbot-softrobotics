from dataclasses import replace

from qdrant_client import QdrantClient

from backend.pipeline.retrieve import Chunk, ChunkIndex
from ingest.purge_test_docs import purge
from tests.fakes import FakeEmbedder

REAL = Chunk(
    id="wikipedia:en:1:0",
    doc_id="wikipedia:en:1",
    text="Silicone rubber is a polymer.",
    language="eng_Latn",
    source="wikipedia",
    title="Silicone rubber",
    url="https://example.org/1",
    licence="CC BY-SA 4.0",
)
FIXTURE = Chunk(
    id="fixture-en:0",
    doc_id="fixture-en",
    text="Fixture text about silicone moulding. It exists only for tests.",
    language="eng_Latn",
    source="fixture",
    title="Silicone fixture",
    url="https://example.invalid/en",
    licence="test-only",
)


def make_index():
    return ChunkIndex(QdrantClient(location=":memory:"), FakeEmbedder())


def test_purge_removes_the_test_only_documents_and_leaves_the_corpus():
    index = make_index()
    index.add([REAL, FIXTURE, replace(FIXTURE, id="fixture-en:1", text="printing")])
    assert purge(index) == (["fixture-en"], 2)
    assert index.doc_ids() == {"wikipedia:en:1"}
    assert index.non_corpus_doc_ids() == set()


def test_purge_on_a_clean_index_removes_nothing():
    index = make_index()
    index.add([REAL])
    assert purge(index) == ([], 0)
    assert index.doc_ids() == {"wikipedia:en:1"}
