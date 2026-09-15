import httpx
import pytest

from backend.privacy.egress import EgressBlocked
from backend.search.base import SearchError, WebResult
from backend.search.google import SEARCH_URL, GoogleSearch


def search_with(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return GoogleSearch("secret-key", "engine-1", client=client)


def test_returns_titles_links_and_snippets():
    seen = {}

    def handler(request):
        seen["url"] = request.url
        items = [
            {"title": "Soft robotics", "link": "https://example.org/a", "snippet": "Soft  robots\nbend."},
            {"title": "Missing link"},
        ]
        return httpx.Response(200, json={"items": items})

    results = search_with(handler).search("soft robot", max_results=3)
    assert results == [WebResult("Soft robotics", "https://example.org/a", "Soft robots bend.", "google")]
    assert seen["url"].copy_with(query=None) == httpx.URL(SEARCH_URL)
    assert dict(seen["url"].params) == {"key": "secret-key", "cx": "engine-1", "q": "soft robot", "num": "3"}


def test_asks_for_at_most_ten_results():
    seen = {}

    def handler(request):
        seen["num"] = request.url.params["num"]
        return httpx.Response(200, json={})

    assert search_with(handler).search("x", max_results=25) == []
    assert seen["num"] == "10"


def test_error_status_includes_googles_message_but_not_the_key():
    def handler(request):
        body = {"error": {"code": 403, "message": "Project secret-key has no access to Custom Search."}}
        return httpx.Response(403, json=body)

    with pytest.raises(SearchError) as error:
        search_with(handler).search("x", max_results=5)
    assert error.value.code == "web_search_failed"
    assert "403" in str(error.value)
    assert "no access to Custom Search" in str(error.value)
    assert "secret-key" not in str(error.value)


def test_request_failure_does_not_repeat_the_url_with_the_key():
    def handler(request):
        raise httpx.ReadTimeout(f"timed out reading {request.url}", request=request)

    with pytest.raises(SearchError) as error:
        search_with(handler).search("x", max_results=5)
    assert error.value.code == "web_search_failed"
    assert "secret-key" not in str(error.value)
    assert error.value.__cause__ is None


def test_blocked_connection_is_reported_as_egress_blocked():
    def handler(request):
        raise httpx.ConnectError("blocked", request=request) from EgressBlocked("www.googleapis.com", 443)

    with pytest.raises(SearchError) as error:
        search_with(handler).search("x", max_results=5)
    assert error.value.code == "egress_blocked"


def test_body_that_is_not_json_is_a_search_error():
    with pytest.raises(SearchError) as error:
        search_with(lambda request: httpx.Response(200, text="<html>")).search("x", max_results=5)
    assert error.value.code == "web_search_failed"
