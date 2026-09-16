import sys
from pathlib import Path

from ingest.datasheet import main, render, summarize
from ingest.documents import make_document, write_jsonl

SAMPLE_METHODS = Path(__file__).parent / "fixtures" / "methods" / "sample.csv"


def doc(doc_id: str, language: str, source: str, licence: str, text: str):
    return make_document(
        id=doc_id,
        text=text,
        language=language,
        source=source,
        title=doc_id,
        url="https://example.invalid/x",
        licence=licence,
        retrieved_at="2026-09-16",
    )


DOCUMENTS = [
    doc("arxiv:1", "eng_Latn", "arxiv", "arXiv metadata, CC0 1.0", "An abstract about silicone."),
    doc("arxiv:2", "eng_Latn", "arxiv", "arXiv metadata, CC0 1.0", "A second abstract."),
    doc("wikipedia:ko:1", "kor_Hang", "wikipedia", "CC BY-SA 4.0", "소프트 로보틱스는 로봇공학의 분야이다."),
]


def test_counts_every_supported_language_including_the_empty_ones():
    summary = summarize(DOCUMENTS, unverified_rows=9, generated_on="2026-09-16")
    assert summary.total == 3
    assert summary.by_language["eng_Latn"] == 2
    assert summary.by_language["kor_Hang"] == 1
    assert summary.by_language["tam_Taml"] == 0
    assert "tam_Taml" in summary.missing_languages


def test_counts_sources_and_licences():
    summary = summarize(DOCUMENTS, unverified_rows=9, generated_on="2026-09-16")
    assert summary.by_source == {"arxiv": 2, "wikipedia": 1}
    assert summary.by_licence == {"CC BY-SA 4.0": 1, "arXiv metadata, CC0 1.0": 2}


def test_the_tables_use_the_counted_numbers():
    text = render(summarize(DOCUMENTS, unverified_rows=9, generated_on="2026-09-16"))
    assert "Documents: 3" in text
    assert "| English | eng_Latn | 2 |" in text
    assert "| Tamil | tam_Taml | 0 |" in text
    assert "| wikipedia | 1 |" in text
    assert "| arXiv metadata, CC0 1.0 | 2 |" in text


def test_unchecked_rows_and_gaps_are_stated():
    text = render(summarize(DOCUMENTS, unverified_rows=9, generated_on="2026-09-16"))
    assert "not checked against a source: 9" in text
    assert "No documents at all in:" in text
    assert "Tamil (tam_Taml)" in text
    assert "lead sections only" in text


def test_main_writes_the_datasheet(tmp_path, monkeypatch, capsys):
    documents = tmp_path / "knowledge-base.jsonl"
    write_jsonl(documents, DOCUMENTS)
    out = tmp_path / "DATASHEET.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "datasheet",
            "--documents",
            str(documents),
            "--methods",
            str(SAMPLE_METHODS),
            "--out",
            str(out),
        ],
    )
    main()
    text = out.read_text(encoding="utf-8")
    assert "| English | eng_Latn | 2 |" in text
    assert "not checked against a source: 1" in text
    assert "from 3 documents" in capsys.readouterr().out
