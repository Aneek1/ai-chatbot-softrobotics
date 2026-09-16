from pathlib import Path

from ingest.build_kb import build
from ingest.documents import IngestError, make_document, read_jsonl, write_jsonl

FIXTURE_METHODS = Path(__file__).parent / "fixtures" / "methods" / "sample.csv"


def doc(doc_id: str, text: str, source: str = "arxiv"):
    return make_document(
        id=doc_id,
        text=text,
        language="eng_Latn",
        source=source,
        title=doc_id,
        url="https://example.invalid/x",
        licence="test-only",
        retrieved_at="2026-09-16",
    )


class FakeArxiv:
    def __init__(self, documents=(), error=None):
        self._documents = list(documents)
        self._error = error
        self.queries: list[tuple[str, int]] = []

    def search(self, query, max_results, retrieved_at=None):
        self.queries.append((query, max_results))
        if self._error is not None:
            raise self._error
        return self._documents[:max_results]


class FakeWikipedia:
    def __init__(self, documents=(), error=None):
        self._documents = list(documents)
        self._error = error

    def articles(self, titles, retrieved_at=None):
        if self._error is not None:
            raise self._error
        return list(self._documents)


def test_merges_the_three_sources():
    arxiv = FakeArxiv([doc("arxiv:1", "An abstract about silicone.")])
    result = build(
        methods_path=FIXTURE_METHODS,
        arxiv_client=arxiv,
        wikipedia_client=FakeWikipedia([doc("wikipedia:en:1", "An article.", source="wikipedia")]),
        queries=('all:"soft robotics"',),
        articles=("Soft robotics",),
        per_query=5,
        retrieved_at="2026-09-16",
    )
    assert [d.id for d in result.documents] == ["csv:test-moulding", "arxiv:1", "wikipedia:en:1"]
    assert arxiv.queries == [('all:"soft robotics"', 5)]
    assert result.unverified_rows == 1
    assert result.errors == []


def test_duplicates_across_sources_are_dropped():
    same = "Soft robots are made of elastomers."
    result = build(
        methods_path=FIXTURE_METHODS,
        arxiv_client=FakeArxiv([doc("arxiv:1", same), doc("arxiv:2", same)]),
        wikipedia_client=FakeWikipedia(),
        queries=("q",),
        articles=(),
        per_query=5,
        retrieved_at="2026-09-16",
    )
    assert [d.id for d in result.documents] == ["csv:test-moulding", "arxiv:1"]
    assert result.duplicates == [("arxiv:2", "arxiv:1")]


def test_a_failing_source_does_not_stop_the_build():
    result = build(
        methods_path=FIXTURE_METHODS,
        arxiv_client=FakeArxiv(error=IngestError("arXiv returned 503")),
        wikipedia_client=FakeWikipedia([doc("wikipedia:en:1", "An article.", source="wikipedia")]),
        queries=("q",),
        articles=("Soft robotics",),
        per_query=5,
        retrieved_at="2026-09-16",
    )
    assert [d.id for d in result.documents] == ["csv:test-moulding", "wikipedia:en:1"]
    assert result.errors == ["arxiv (q): IngestError: arXiv returned 503"]


def test_a_missing_methods_file_is_reported():
    result = build(
        methods_path=Path("does-not-exist.csv"),
        arxiv_client=FakeArxiv(),
        wikipedia_client=FakeWikipedia(),
        queries=(),
        articles=(),
        per_query=5,
        retrieved_at="2026-09-16",
    )
    assert result.documents == []
    assert result.errors[0].startswith("csv: FileNotFoundError")


def test_the_result_is_written_as_jsonl(tmp_path):
    result = build(
        methods_path=FIXTURE_METHODS,
        arxiv_client=FakeArxiv([doc("arxiv:1", "An abstract.")]),
        wikipedia_client=FakeWikipedia(),
        queries=("q",),
        articles=(),
        per_query=5,
        retrieved_at="2026-09-16",
    )
    path = tmp_path / "knowledge-base.jsonl"
    assert write_jsonl(path, result.documents) == 2
    assert [d.id for d in read_jsonl(path)] == ["csv:test-moulding", "arxiv:1"]
