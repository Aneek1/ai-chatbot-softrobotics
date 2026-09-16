from pathlib import Path

import pytest

from ingest.documents import IngestError
from ingest.methods_csv import build_documents, load_methods, read_rows

FIXTURE = Path(__file__).parent / "fixtures" / "methods" / "sample.csv"
REPO_METHODS = Path(__file__).resolve().parents[1] / "data" / "methods.csv"


def rows(**overrides):
    row = {
        "method": "Moulding",
        "description": "Casting an elastomer in a mould.",
        "materials": "silicone",
        "process": "Mix, pour, cure, demould.",
        "language": "eng_Latn",
        "source_title": "Fixture paper",
        "source_url": "https://example.invalid/paper",
        "licence": "CC BY 4.0",
        "status": "verified",
        "checked_by": "Test Curator",
        "checked_on": "2026-09-16",
        "notes": "",
    }
    row.update(overrides)
    return [row]


def test_a_verified_row_becomes_a_document():
    documents, unverified = build_documents(rows(), "2026-09-16")
    assert unverified == []
    document = documents[0]
    assert document.id == "csv:moulding"
    assert document.source == "csv"
    assert document.language == "eng_Latn"
    assert document.url == "https://example.invalid/paper"
    assert document.licence == "CC BY 4.0"
    assert document.revision == "checked 2026-09-16 by Test Curator"
    assert "Materials: silicone." in document.text
    assert "Process: Mix, pour, cure, demould." in document.text
    assert "Source: Fixture paper (https://example.invalid/paper)." in document.text


def test_an_unverified_row_never_becomes_a_document():
    documents, unverified = build_documents(
        rows(status="unverified", source_url="", licence="", checked_by="", checked_on=""),
        "2026-09-16",
    )
    assert documents == []
    assert [row["method"] for row in unverified] == ["Moulding"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"source_url": ""}, "verified rows need 'source_url'"),
        ({"licence": ""}, "verified rows need 'licence'"),
        ({"checked_by": ""}, "verified rows need 'checked_by'"),
        ({"source_url": "doi:10.1000/xyz"}, "source_url must be an http"),
        ({"checked_on": "16-09-2026"}, "checked_on must be YYYY-MM-DD"),
        ({"status": "maybe"}, "status must be one of"),
    ],
)
def test_a_row_that_cannot_be_checked_is_an_error(overrides, message):
    with pytest.raises(IngestError, match=message):
        build_documents(rows(**overrides), "2026-09-16")


def test_a_missing_column_is_an_error():
    with pytest.raises(IngestError, match="missing columns"):
        read_rows("method,description\nMoulding,x\n")


def test_the_fixture_file_has_one_verified_and_one_unverified_row():
    documents, unverified = load_methods(FIXTURE, "2026-09-16")
    assert [d.id for d in documents] == ["csv:test-moulding"]
    assert [row["method"] for row in unverified] == ["Test printing"]


def test_the_repo_csv_holds_no_verified_rows_yet():
    documents, unverified = load_methods(REPO_METHODS, "2026-09-16")
    assert documents == []
    assert unverified == []
