from pathlib import Path

import httpx
import pytest

from ingest.arxiv import API_URL, ArxivClient, parse_feed
from ingest.documents import IngestError

FIXTURE = Path(__file__).parent / "fixtures" / "arxiv" / "soft-robotics.xml"


def feed() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def client_with(handler, **kwargs) -> ArxivClient:
    return ArxivClient(client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs)


def test_parses_the_recorded_feed():
    documents = parse_feed(feed(), "2026-09-16")
    assert [d.id for d in documents] == ["arxiv:2609.14376", "arxiv:2609.12258"]
    first = documents[0]
    assert first.title == "Orientation Control of Soft Robots via Adiabatic Spectral Submanifolds"
    assert first.url == "https://arxiv.org/abs/2609.14376v1"
    assert first.revision == "v1"
    assert first.source == "arxiv"
    assert first.language == "eng_Latn"
    assert first.script == "Latn"
    assert first.licence == "arXiv metadata, CC0 1.0"
    assert first.retrieved_at == "2026-09-16"
    assert first.text.startswith(
        "Orientation Control of Soft Robots via Adiabatic Spectral Submanifolds. Soft robots are"
    )
    assert "\n" not in first.text


def test_an_entry_without_an_abstract_is_skipped():
    xml = (
        '<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
        "<id>http://arxiv.org/abs/2609.00001v2</id><title>Only a title</title>"
        "</entry></feed>"
    )
    assert parse_feed(xml, "2026-09-16") == []


def test_an_identifier_without_a_version():
    xml = (
        '<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
        "<id>http://arxiv.org/abs/cs/0701001</id><title>Old style</title>"
        "<summary>An abstract.</summary></entry></feed>"
    )
    document = parse_feed(xml, "2026-09-16")[0]
    assert document.id == "arxiv:cs/0701001"
    assert document.revision == ""
    assert document.url == "https://arxiv.org/abs/cs/0701001"


def test_search_sends_the_documented_parameters():
    seen = {}

    def handler(request):
        seen["url"] = request.url
        seen["user_agent"] = request.headers["user-agent"]
        return httpx.Response(200, text=feed())

    documents = client_with(handler).search('all:"soft robotics"', 2, retrieved_at="2026-09-16")
    assert len(documents) == 2
    assert str(seen["url"]).startswith(API_URL)
    params = seen["url"].params
    assert params["search_query"] == 'all:"soft robotics"'
    assert params["max_results"] == "2"
    assert params["sortBy"] == "submittedDate"
    assert seen["user_agent"].startswith("softrobotics-assistant/")


def test_requests_are_spaced_out():
    waited = []

    def handler(request):
        return httpx.Response(200, text=feed())

    client = client_with(handler, pause=3.0, sleep=waited.append)
    client.search("a", 1, retrieved_at="2026-09-16")
    client.search("b", 1, retrieved_at="2026-09-16")
    assert waited == [3.0]


def test_an_error_status_is_reported():
    def handler(request):
        return httpx.Response(503, text="temporarily unavailable")

    with pytest.raises(IngestError, match="arXiv returned 503"):
        client_with(handler).search("a", 1, retrieved_at="2026-09-16")
