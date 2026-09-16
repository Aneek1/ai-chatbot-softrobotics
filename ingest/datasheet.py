"""Write DATASHEET.md from the built knowledge base (spec 6.3).

Usage: PYTHONUTF8=1 uv run python -m ingest.datasheet
"""

import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from backend.pipeline.languages import SUPPORTED, display_name
from ingest.documents import Document, read_jsonl, today
from ingest.methods_csv import load_methods

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS = REPO_ROOT / "data" / "knowledge-base.jsonl"
DEFAULT_METHODS = REPO_ROOT / "data" / "methods.csv"
DEFAULT_OUT = REPO_ROOT / "DATASHEET.md"


@dataclass(frozen=True)
class Summary:
    total: int
    by_language: dict[str, int]
    by_source: dict[str, int]
    by_licence: dict[str, int]
    unverified_rows: int
    generated_on: str

    @property
    def missing_languages(self) -> list[str]:
        return [label for label, count in self.by_language.items() if count == 0]


def summarize(documents: Sequence[Document], unverified_rows: int, generated_on: str) -> Summary:
    languages = Counter(document.language for document in documents)
    by_language = {label: languages.get(label, 0) for label in SUPPORTED}
    for label, count in sorted(languages.items()):
        by_language.setdefault(label, count)
    return Summary(
        total=len(documents),
        by_language=by_language,
        by_source=dict(sorted(Counter(d.source for d in documents).items())),
        by_licence=dict(sorted(Counter(d.licence for d in documents).items())),
        unverified_rows=unverified_rows,
        generated_on=generated_on,
    )


def render(summary: Summary) -> str:
    lines = [
        "# Knowledge base datasheet",
        "",
        f"Written by `ingest/datasheet.py` on {summary.generated_on} from",
        "`data/knowledge-base.jsonl`, the file `ingest/build_kb.py` builds. Every number here is",
        "counted from that file.",
        "",
        f"Documents: {summary.total}",
        "",
        "## Documents by language",
        "",
        "| Language | Label | Documents |",
        "|---|---|---|",
    ]
    lines += [
        f"| {display_name(label)} | {label} | {count} |" for label, count in summary.by_language.items()
    ]
    lines += ["", "## Documents by source", "", "| Source | Documents |", "|---|---|"]
    lines += [f"| {source} | {count} |" for source, count in summary.by_source.items()]
    lines += ["", "## Licences", "", "| Licence | Documents |", "|---|---|"]
    lines += [f"| {licence} | {count} |" for licence, count in summary.by_licence.items()]
    lines += [
        "",
        "Each document records its `source`, `url`, `licence` and `retrieved_at`, and, where the",
        "source has one, its `revision`: the arXiv version or the Wikipedia revision id.",
        "",
        "## Curated rows",
        "",
        f"Rows in `data/methods.csv` not checked against a source: {summary.unverified_rows}.",
        "Those rows produce no documents. They are never retrieved, never cited and never sent to an",
        "answer model. A row becomes a document only when it carries a source title, an http(s) link",
        "to that source, a licence, and the name and date of the person who checked it.",
        "`docs/knowledge-base-curation.md` has the rules.",
        "",
        "## Known gaps",
        "",
    ]
    if summary.missing_languages:
        names = ", ".join(f"{display_name(label)} ({label})" for label in summary.missing_languages)
        lines.append(f"- No documents at all in: {names}. Questions in those languages are answered")
        lines.append("  from sources in other languages, or not at all.")
    lines += [
        "- Wikipedia documents are lead sections only, not whole articles, and only for the languages",
        "  that have an article linked from the English one.",
        "- arXiv documents are titles and abstracts in English. Full texts are not fetched.",
        "- Chinese comes from the Simplified variant of one Wikipedia, so Traditional Chinese",
        "  (`zho_Hant`) has no documents of its own.",
        "- Nothing here is translated, and no text is generated. What a source said is what is stored.",
        "",
        "## How it was collected",
        "",
        "`ingest/build_kb.py` reads `data/methods.csv`, queries the arXiv API and the Wikipedia API,",
        "drops exact duplicates by hash of the normalized text, and writes the JSONL. It runs outside",
        "the backend process, so the egress guard does not apply to it; it connects to",
        "`export.arxiv.org` and the `*.wikipedia.org` API hosts.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write DATASHEET.md from the built knowledge base")
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--methods", type=Path, default=DEFAULT_METHODS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    documents = read_jsonl(args.documents)
    _, unverified = load_methods(args.methods)
    text = render(summarize(documents, len(unverified), today()))
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {args.out} from {len(documents)} documents")


if __name__ == "__main__":
    main()
