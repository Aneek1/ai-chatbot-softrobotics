import json

import pytest
from qdrant_client import QdrantClient

from backend.pipeline.retrieve import ChunkIndex
from ingest.documents import (
    Document,
    IngestError,
    make_document,
    read_jsonl,
    to_json_line,
    write_jsonl,
)
from ingest.load_jsonl import load_documents
from tests.fakes import FakeEmbedder


def doc(**overrides) -> Document:
    fields = {
        "id": "arxiv:1234.5678",
        "text": "Soft actuators are cast in silicone.",
        "language": "eng_Latn",
        "script": "Latn",
        "source": "arxiv",
        "title": "Soft actuators",
        "url": "https://example.invalid/paper",
        "licence": "arXiv metadata, CC0 1.0",
        "retrieved_at": "2026-09-16",
        "revision": "v1",
    }
    fields.update(overrides)
    return Document(**fields)


def test_make_document_normalizes_the_text_and_records_the_script():
    document = make_document(
        id="wikipedia:ko:2595374",
        text="소프트\n로보틱스는  로봇공학의 하위 분야이다.",
        language="kor_Hang",
        source="wikipedia",
        title="  소프트 로보틱스 ",
        url="https://ko.wikipedia.org/wiki/x",
        licence="CC BY-SA 4.0",
        retrieved_at="2026-09-16",
        revision="40744485",
    )
    assert document.text == "소프트 로보틱스는 로봇공학의 하위 분야이다."
    assert document.title == "소프트 로보틱스"
    assert document.script == "Hang"


def test_json_line_round_trip():
    document = doc()
    assert Document(**json.loads(to_json_line(document))) == document


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"licence": ""}, "'licence' must not be empty"),
        ({"source": "blog"}, "source must be one of"),
        ({"url": "example.invalid/paper"}, "url must be an http"),
        ({"retrieved_at": "16-09-2026"}, "retrieved_at must be YYYY-MM-DD"),
    ],
)
def test_a_document_without_provenance_is_rejected(overrides, message):
    with pytest.raises(IngestError, match=message):
        doc(**overrides)


def test_write_and_read_jsonl(tmp_path):
    path = tmp_path / "kb" / "knowledge-base.jsonl"
    assert write_jsonl(path, [doc(), doc(id="arxiv:2222.3333")]) == 2
    assert [d.id for d in read_jsonl(path)] == ["arxiv:1234.5678", "arxiv:2222.3333"]


def test_read_jsonl_names_a_broken_line(tmp_path):
    path = tmp_path / "knowledge-base.jsonl"
    path.write_text(to_json_line(doc()) + "\n{not json\n", encoding="utf-8")
    with pytest.raises(IngestError, match="line 2: invalid JSON"):
        read_jsonl(path)


def test_written_documents_load_into_the_index(tmp_path):
    path = tmp_path / "knowledge-base.jsonl"
    write_jsonl(
        path,
        [
            doc(),
            doc(
                id="wikipedia:en:1",
                source="wikipedia",
                language="ind_Latn",
                licence="CC BY-SA 4.0",
                url="https://example.invalid/id",
            ),
        ],
    )
    index = ChunkIndex(QdrantClient(location=":memory:"), FakeEmbedder())
    added = load_documents(path.read_text(encoding="utf-8").splitlines(), index, count=len)
    assert added == 2
    assert index.count_by_language() == {"eng_Latn": 1, "ind_Latn": 1}
