import hashlib

from backend.pipeline.normalize import normalize
from ingest.dedup import dedupe, text_hash
from ingest.documents import make_document


def doc(doc_id: str, text: str):
    return make_document(
        id=doc_id,
        text=text,
        language="eng_Latn",
        source="arxiv",
        title="t",
        url="https://example.invalid/x",
        licence="test-only",
        retrieved_at="2026-09-16",
    )


def test_text_hash_is_the_hash_of_the_normalized_text():
    expected = hashlib.sha256(normalize("Soft robots bend.").encode("utf-8")).hexdigest()
    assert text_hash("  Soft\nrobots bend. ") == expected


def test_the_same_text_is_kept_once():
    result = dedupe([doc("a", "Soft robots bend."), doc("b", "Soft robots bend.")])
    assert [d.id for d in result.kept] == ["a"]
    assert result.dropped == [("b", "a")]


def test_whitespace_differences_are_the_same_document():
    result = dedupe([doc("a", "Soft robots bend."), doc("b", "Soft   robots\nbend.")])
    assert [d.id for d in result.kept] == ["a"]


def test_different_text_is_kept():
    result = dedupe([doc("a", "Soft robots bend."), doc("b", "Rigid robots do not.")])
    assert [d.id for d in result.kept] == ["a", "b"]
    assert result.dropped == []


def test_a_repeated_id_is_dropped_even_with_different_text():
    result = dedupe([doc("a", "Soft robots bend."), doc("a", "Rigid robots do not.")])
    assert [d.id for d in result.kept] == ["a"]
    assert result.dropped == [("a", "a")]
