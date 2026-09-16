"""The knowledge-base document record (spec 6.1), shared by every source builder."""

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from backend.pipeline.normalize import normalize, script_profile

SOURCES = ("csv", "arxiv", "wikipedia")
_REQUIRED = ("id", "text", "language", "source", "title", "url", "licence", "retrieved_at")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class IngestError(ValueError):
    """A source file or API response the knowledge-base build cannot use."""


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    language: str
    script: str
    source: str
    title: str
    url: str
    licence: str
    retrieved_at: str
    revision: str = ""

    def __post_init__(self) -> None:
        for name in _REQUIRED:
            if not getattr(self, name):
                raise IngestError(f"{self.id or 'document'}: '{name}' must not be empty")
        if self.source not in SOURCES:
            raise IngestError(f"{self.id}: source must be one of {', '.join(SOURCES)}, got '{self.source}'")
        if not self.url.startswith(("http://", "https://")):
            raise IngestError(f"{self.id}: url must be an http(s) link, got '{self.url}'")
        if not _DATE.match(self.retrieved_at):
            raise IngestError(f"{self.id}: retrieved_at must be YYYY-MM-DD, got '{self.retrieved_at}'")


def today() -> str:
    return date.today().isoformat()


def make_document(
    *,
    id: str,
    text: str,
    language: str,
    source: str,
    title: str,
    url: str,
    licence: str,
    retrieved_at: str,
    revision: str = "",
) -> Document:
    """Normalize the text the same way questions are normalized, and record its dominant script."""
    clean = normalize(text)
    return Document(
        id=id,
        text=clean,
        language=language,
        script=script_profile(clean).dominant or "",
        source=source,
        title=normalize(title),
        url=url,
        licence=licence,
        retrieved_at=retrieved_at,
        revision=revision,
    )


def to_json_line(document: Document) -> str:
    return json.dumps(asdict(document), ensure_ascii=False, sort_keys=True)


def write_jsonl(path: Path, documents: Iterable[Document]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for document in documents:
            handle.write(to_json_line(document) + "\n")
            written += 1
    return written


def read_jsonl(path: Path) -> list[Document]:
    documents: list[Document] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as error:
            raise IngestError(f"{path}: line {number}: invalid JSON: {error.msg}") from error
        try:
            documents.append(Document(**data))
        except TypeError as error:
            raise IngestError(f"{path}: line {number}: {error}") from error
    return documents
