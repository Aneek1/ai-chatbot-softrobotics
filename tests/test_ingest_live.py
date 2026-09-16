"""Checks against the real arXiv and Wikipedia APIs.

Marked slow: excluded from the default run and from CI. Run them when a fixture looks stale:
    uv run pytest tests/test_ingest_live.py -m slow -v
"""

import pytest

from ingest.arxiv import ArxivClient
from ingest.wikipedia import WIKI_LANGUAGES, WikipediaClient

pytestmark = pytest.mark.slow


def test_arxiv_returns_documents():
    documents = ArxivClient().search('all:"soft robotics"', 2)
    assert len(documents) == 2
    for document in documents:
        assert document.id.startswith("arxiv:")
        assert document.url.startswith("https://arxiv.org/abs/")
        assert document.text
        assert document.licence == "arXiv metadata, CC0 1.0"


def test_wikipedia_returns_the_english_article_and_its_language_links():
    document, links = WikipediaClient().page("en", "Soft robotics")
    assert document is not None
    assert document.id.startswith("wikipedia:en:")
    assert document.revision.isdigit()
    assert set(links) <= set(WIKI_LANGUAGES) - {"en"}
