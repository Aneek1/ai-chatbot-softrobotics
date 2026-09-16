from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from backend.pipeline.retrieve import ChunkIndex
from ingest.load_jsonl import load_documents
from tests.fakes import FakeEmbedder, word_count

FIXTURE = Path(__file__).parent / "fixtures" / "documents.jsonl"


def test_loads_fixture_documents():
    index = ChunkIndex(QdrantClient(location=":memory:"), FakeEmbedder())
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    added = load_documents(lines, index, count=len)
    assert added == 3
    assert index.count_by_language() == {"eng_Latn": 1, "ind_Latn": 1, "zho_Hans": 1}


def test_long_document_becomes_several_chunks():
    index = ChunkIndex(QdrantClient(location=":memory:"), FakeEmbedder())
    doc = (
        '{"id": "long", "text": "one two three. four five six. seven eight nine.", "language": "eng_Latn", '
        '"source": "fixture", "title": "t", "url": "https://example.invalid", "licence": "test-only"}'
    )
    added = load_documents([doc], index, count=word_count, max_tokens=4, overlap_tokens=0)
    assert added == 3
    hits = index.search("silicone", top_k=10)
    assert sorted(h.chunk.id for h in hits) == ["long:0", "long:1", "long:2"]


def test_missing_field_names_the_line():
    index = ChunkIndex(QdrantClient(location=":memory:"), FakeEmbedder())
    with pytest.raises(ValueError, match="line 1: missing 'licence'"):
        incomplete = (
            '{"id": "x", "text": "t", "language": "eng_Latn", '
            '"source": "s", "title": "t", "url": "u"}'
        )
        load_documents([incomplete], index, count=len)


def test_duplicate_id_names_the_line():
    index = ChunkIndex(QdrantClient(location=":memory:"), FakeEmbedder())
    first = (
        '{"id": "dup", "text": "hello world", "language": "eng_Latn", '
        '"source": "s", "title": "t1", "url": "u", "licence": "test-only"}'
    )
    second = first.replace("hello", "goodbye").replace('"t1"', '"t2"')
    with pytest.raises(ValueError, match="line 3: duplicate id 'dup'"):
        load_documents([first, "", second], index, count=len)
    assert index.count_by_language() == {}


def test_invalid_json_names_the_line():
    index = ChunkIndex(QdrantClient(location=":memory:"), FakeEmbedder())
    valid = (
        '{"id": "ok", "text": "hello", "language": "eng_Latn", '
        '"source": "s", "title": "t", "url": "u", "licence": "test-only"}'
    )
    broken = '{"id": "x", "text": "t", not valid json'
    with pytest.raises(ValueError, match=r"^line 3: invalid JSON: Expecting property name"):
        load_documents([valid, "", broken], index, count=len)
    assert index.count_by_language() == {}


@pytest.mark.parametrize(("line", "kind"), [("42", "int"), ("null", "NoneType"), ('["a"]', "list")])
def test_non_object_line_names_the_line(line, kind):
    index = ChunkIndex(QdrantClient(location=":memory:"), FakeEmbedder())
    with pytest.raises(ValueError, match=f"^line 2: expected a JSON object, got {kind}$"):
        load_documents(["", line], index, count=len)


@pytest.mark.parametrize("value", ["[1, 2]", "7", "null"])
def test_non_string_field_names_the_line(value):
    index = ChunkIndex(QdrantClient(location=":memory:"), FakeEmbedder())
    doc = (
        f'{{"id": {value}, "text": "hello", "language": "eng_Latn", '
        '"source": "s", "title": "t", "url": "u", "licence": "test-only"}'
    )
    with pytest.raises(ValueError, match="^line 1: 'id' must be a string$"):
        load_documents([doc], index, count=len)


def test_blank_lines_are_skipped():
    index = ChunkIndex(QdrantClient(location=":memory:"), FakeEmbedder())
    assert load_documents(["", "   "], index, count=len) == 0


def test_provenance_fields_are_kept():
    index = ChunkIndex(QdrantClient(location=":memory:"), FakeEmbedder())
    doc = (
        '{"id": "w1", "text": "hello", "language": "eng_Latn", "source": "wikipedia", '
        '"title": "t", "url": "https://example.invalid", "licence": "CC BY-SA 4.0", '
        '"script": "Latn", "retrieved_at": "2026-09-16", "revision": "1368852135"}'
    )
    assert load_documents([doc], index, count=len) == 1
    chunk = index.search("hello", top_k=1)[0].chunk
    assert chunk.script == "Latn"
    assert chunk.retrieved_at == "2026-09-16"
    assert chunk.revision == "1368852135"


def test_script_is_derived_when_the_document_does_not_give_one():
    index = ChunkIndex(QdrantClient(location=":memory:"), FakeEmbedder())
    doc = (
        '{"id": "k1", "text": "소프트 로봇", "language": "kor_Hang", "source": "wikipedia", '
        '"title": "t", "url": "https://example.invalid", "licence": "CC BY-SA 4.0"}'
    )
    assert load_documents([doc], index, count=len) == 1
    chunk = index.search("소프트", top_k=1)[0].chunk
    assert chunk.script == "Hang"
    assert chunk.retrieved_at == ""
    assert chunk.revision == ""


@pytest.mark.parametrize("name", ["script", "retrieved_at", "revision"])
def test_non_string_provenance_field_names_the_line(name):
    index = ChunkIndex(QdrantClient(location=":memory:"), FakeEmbedder())
    doc = (
        '{"id": "x", "text": "hello", "language": "eng_Latn", "source": "s", '
        '"title": "t", "url": "u", "licence": "test-only", ' + f'"{name}": 7' + "}"
    )
    with pytest.raises(ValueError, match=f"^line 1: '{name}' must be a string$"):
        load_documents([doc], index, count=len)
