"""Titles and abstracts from the arXiv API. arXiv metadata is CC0; full texts are not fetched."""

import time
from collections.abc import Callable
from xml.etree import ElementTree

import httpx

from backend.pipeline.normalize import normalize
from ingest.documents import Document, IngestError, make_document, today

API_URL = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
LICENCE = "arXiv metadata, CC0 1.0"
USER_AGENT = "softrobotics-assistant/0.1 (knowledge-base build)"
LANGUAGE = "eng_Latn"  # arXiv abstracts in this subject area are English


def _identifier(entry_id: str) -> tuple[str, str]:
    """('2609.14376v1') -> ('2609.14376', 'v1'); an id with no version keeps an empty version."""
    name = entry_id.strip().rsplit("/abs/", 1)[-1]
    base, separator, version = name.rpartition("v")
    if separator and version.isdigit():
        return base, f"v{version}"
    return name, ""


def parse_feed(xml_text: str, retrieved_at: str) -> list[Document]:
    root = ElementTree.fromstring(xml_text)
    documents: list[Document] = []
    for entry in root.findall(f"{ATOM}entry"):
        entry_id = (entry.findtext(f"{ATOM}id") or "").strip()
        title = normalize(entry.findtext(f"{ATOM}title") or "")
        summary = normalize(entry.findtext(f"{ATOM}summary") or "")
        if not entry_id or not title or not summary:
            continue
        paper_id, version = _identifier(entry_id)
        documents.append(
            make_document(
                id=f"arxiv:{paper_id}",
                text=f"{title}. {summary}",
                language=LANGUAGE,
                source="arxiv",
                title=title,
                url=f"https://arxiv.org/abs/{paper_id}{version}",
                licence=LICENCE,
                retrieved_at=retrieved_at,
                revision=version,
            )
        )
    return documents


class ArxivClient:
    """One request every `pause` seconds, which is what the arXiv API terms of use ask for."""

    def __init__(
        self,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
        pause: float = 3.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._client = client or httpx.Client(timeout=timeout)
        self._pause = pause
        self._sleep = sleep
        self._requested = False

    def search(self, query: str, max_results: int, retrieved_at: str | None = None) -> list[Document]:
        if self._requested:
            self._sleep(self._pause)
        self._requested = True
        params = {
            "search_query": query,
            "start": "0",
            "max_results": str(max_results),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        try:
            response = self._client.get(API_URL, params=params, headers={"User-Agent": USER_AGENT})
        except httpx.HTTPError as exc:
            raise IngestError(f"arXiv request failed ({type(exc).__name__})") from exc
        if response.status_code != 200:
            raise IngestError(f"arXiv returned {response.status_code}")
        try:
            return parse_feed(response.text, retrieved_at or today())
        except ElementTree.ParseError as exc:
            raise IngestError(f"arXiv sent a response that is not XML: {exc}") from exc
