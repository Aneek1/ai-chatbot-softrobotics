import re
import socket
import threading
import tomllib
from pathlib import Path

import httpx
import pytest

from backend.privacy.egress import EgressBlocked
from backend.search.base import SearchError, WebResult
from backend.search.duckduckgo import SEARCH_URL, USER_AGENT, DuckDuckGoSearch, parse_results

ROOT = Path(__file__).resolve().parents[1]

# Trimmed from a real response of https://html.duckduckgo.com/html/ (2026-09-16), plus an advert
# and a redirect link, which the endpoint also returns.
RESULT_PAGE = """<html><body><div id="links" class="results">
<div class="result results_links results_links_deep web-result ">
  <div class="links_main links_deep result__body">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a" href="https://example.org/casting">Silicone <b>casting</b></a>
    </h2>
    <div class="result__extras">
      <a class="result__url" href="https://example.org/casting">example.org/casting</a>
    </div>
    <a class="result__snippet" href="https://example.org/casting">Pour <b>Ecoflex</b> into a
      printed mould &amp; degas it.</a>
  </div>
</div>
<div class="result results_links results_links_deep result--ad">
  <div class="links_main links_deep result__body">
    <h2 class="result__title">
      <a class="result__a" href="https://duckduckgo.com/y.js?ad_domain=shop.example">Buy silicone</a>
    </h2>
    <a class="result__snippet" href="https://duckduckgo.com/y.js?ad_domain=shop.example">Advert</a>
  </div>
</div>
<div class="result results_links results_links_deep web-result ">
  <div class="links_main links_deep result__body">
    <h2 class="result__title">
      <a class="result__a"
         href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.net%2Fgel&amp;rut=1">Gel grippers</a>
    </h2>
    <a class="result__snippet"
       href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.net%2Fgel">Gels swell.</a>
  </div>
</div>
<div class="result results_links results_links_deep web-result ">
  <div class="links_main links_deep result__body">
    <h2 class="result__title"><a class="result__a" href="javascript:alert(1)">Script link</a></h2>
  </div>
</div>
</div></body></html>"""

EXPECTED = [
    WebResult(
        "Silicone casting",
        "https://example.org/casting",
        "Pour Ecoflex into a printed mould & degas it.",
        "duckduckgo",
    ),
    WebResult("Gel grippers", "https://example.net/gel", "Gels swell.", "duckduckgo"),
]


def search_with(handler):
    return DuckDuckGoSearch(client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_parses_titles_links_and_snippets_and_drops_adverts():
    assert parse_results(RESULT_PAGE) == EXPECTED


def test_posts_once_to_the_html_endpoint_and_never_opens_result_pages():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, text=RESULT_PAGE)

    results = search_with(handler).search("silicone casting", max_results=5)
    assert results == EXPECTED
    assert [(r.method, str(r.url)) for r in requests] == [("POST", SEARCH_URL)]
    assert requests[0].headers["User-Agent"] == USER_AGENT
    assert requests[0].content == b"q=silicone+casting&b=&l=wt-wt"


def test_max_results_limits_the_list():
    results = search_with(lambda request: httpx.Response(200, text=RESULT_PAGE)).search("x", max_results=1)
    assert results == EXPECTED[:1]


def test_refusal_status_is_a_search_error():
    # DuckDuckGo answers 202 with no results when it declines a request.
    with pytest.raises(SearchError) as error:
        search_with(lambda request: httpx.Response(202, text="")).search("x", max_results=5)
    assert error.value.code == "web_search_failed"
    assert "202" in str(error.value)


def test_timeout_is_a_search_error():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(SearchError) as error:
        search_with(handler).search("x", max_results=5)
    assert error.value.code == "web_search_failed"


def test_blocked_connection_is_reported_as_egress_blocked():
    def handler(request):
        raise httpx.ConnectError("blocked", request=request) from EgressBlocked("html.duckduckgo.com", 443)

    with pytest.raises(SearchError) as error:
        search_with(handler).search("x", max_results=5)
    assert error.value.code == "egress_blocked"


def test_proxy_environment_variables_are_ignored(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.invalid:3128")
    assert DuckDuckGoSearch().client.trust_env is False


class FakeSocksProxy:
    """Accepts one SOCKS5 client, records its CONNECT request, then hangs up."""

    def __init__(self):
        self.server = socket.socket()
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(1)
        self.port = self.server.getsockname()[1]
        self.request = b""
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        connection, _ = self.server.accept()
        with connection:
            connection.recv(3)  # version 5, one method, no authentication
            connection.sendall(b"\x05\x00")
            self.request = connection.recv(262)

    def close(self):
        self.server.close()
        self.thread.join(5)


def test_socks5h_proxy_receives_the_host_name_so_dns_goes_through_the_proxy(monkeypatch):
    looked_up = []
    real_getaddrinfo = socket.getaddrinfo

    def recording_getaddrinfo(host, *args, **kwargs):
        looked_up.append(host)
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", recording_getaddrinfo)
    proxy = FakeSocksProxy()
    try:
        search = DuckDuckGoSearch(timeout=5, proxy=f"socks5h://127.0.0.1:{proxy.port}")
        with pytest.raises(SearchError):
            search.search("silicone", max_results=5)
    finally:
        proxy.close()
    # CONNECT with address type 3 (domain name) carries the name to the proxy unresolved.
    assert proxy.request.startswith(b"\x05\x01\x00\x03\x13html.duckduckgo.com")
    assert "html.duckduckgo.com" not in looked_up


def test_backend_uses_no_http_client_the_egress_guard_cannot_see():
    # ddgs uses primp and curl_cffi wraps libcurl: both open sockets outside Python's socket module.
    pattern = re.compile(r"^\s*(?:import|from)\s+(primp|ddgs|duckduckgo_search|curl_cffi|pycurl)\b", re.M)
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "backend").rglob("*.py")
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
    dependencies = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "dependencies"
    ]
    assert not [d for d in dependencies if d.startswith(("primp", "ddgs", "duckduckgo", "curl"))]
