import json
from pathlib import Path
from urllib.parse import unquote

import httpx
import pytest

from ingest.documents import IngestError
from ingest.wikipedia import WikipediaClient, parse_langlinks, parse_page

FIXTURES = Path(__file__).parent / "fixtures" / "wikipedia"
MISSING = {"batchcomplete": True, "query": {"pages": [{"ns": 0, "title": "Soft robotics", "missing": True}]}}


def payload(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_parses_the_english_page():
    document = parse_page(payload("en-soft-robotics.json"), "en", "2026-09-16")
    assert document.id == "wikipedia:en:50647426"
    assert document.language == "eng_Latn"
    assert document.script == "Latn"
    assert document.source == "wikipedia"
    assert document.licence == "CC BY-SA 4.0"
    assert document.revision == "1368852135"
    assert document.url == "https://en.wikipedia.org/wiki/Soft_robotics"
    assert document.text.startswith("Soft robotics is a subfield of robotics")
    assert "\n" not in document.text


def test_parses_the_korean_page():
    document = parse_page(payload("ko-soft-robotics.json"), "ko", "2026-09-16")
    assert document.id == "wikipedia:ko:2595374"
    assert document.language == "kor_Hang"
    assert document.script == "Hang"
    assert document.revision == "40744485"
    assert unquote(document.url) == "https://ko.wikipedia.org/wiki/소프트_로보틱스"


def test_language_links_are_filtered_to_the_supported_languages():
    links = parse_langlinks(payload("en-soft-robotics.json"))
    assert links == {"id": "Soft robotics", "ko": "소프트 로보틱스"}


def test_a_missing_page_has_no_document():
    assert parse_page(MISSING, "id", "2026-09-16") is None
    assert parse_langlinks(MISSING) == {}


def test_articles_follows_the_language_links():
    calls = []

    def handler(request):
        calls.append((request.url.host, request.headers["user-agent"]))
        if request.url.host == "en.wikipedia.org":
            return httpx.Response(200, json=payload("en-soft-robotics.json"))
        if request.url.host == "ko.wikipedia.org":
            return httpx.Response(200, json=payload("ko-soft-robotics.json"))
        return httpx.Response(200, json=MISSING)

    client = WikipediaClient(
        client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda seconds: None
    )
    documents = client.articles(["Soft robotics"], retrieved_at="2026-09-16")
    assert [d.language for d in documents] == ["eng_Latn", "kor_Hang"]
    assert [host for host, _ in calls] == ["en.wikipedia.org", "id.wikipedia.org", "ko.wikipedia.org"]
    assert all(agent.startswith("softrobotics-assistant/") for _, agent in calls)


def test_chinese_pages_ask_for_the_simplified_variant():
    seen = {}

    def handler(request):
        seen["params"] = request.url.params
        return httpx.Response(200, json=MISSING)

    client = WikipediaClient(client=httpx.Client(transport=httpx.MockTransport(handler)))
    document, links = client.page("zh", "软体机器人", "2026-09-16")
    assert document is None and links == {}
    assert seen["params"]["variant"] == "zh-hans"


def test_an_error_status_is_reported():
    def handler(request):
        return httpx.Response(429, text="slow down")

    client = WikipediaClient(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(IngestError, match="en.wikipedia.org returned 429"):
        client.page("en", "Soft robotics", "2026-09-16")
