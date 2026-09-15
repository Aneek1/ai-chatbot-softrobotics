from html.parser import HTMLParser
from urllib.parse import parse_qs, urlsplit

import httpx

from backend.search.base import SearchError, WebResult, clean_text, request_failure

SEARCH_URL = "https://html.duckduckgo.com/html/"
# On 2026-09-16 the endpoint answered 202 with no results to httpx's default User-Agent.
USER_AGENT = "Mozilla/5.0"
MAX_SNIPPET_CHARS = 500


def result_url(href: str) -> str | None:
    if href.startswith("//"):
        href = "https:" + href
    parts = urlsplit(href)
    host = parts.hostname or ""
    if host == "duckduckgo.com" or host.endswith(".duckduckgo.com"):
        if parts.path == "/l/":
            target = parse_qs(parts.query).get("uddg", [""])[0]
            return result_url(target) if target else None
        return None  # adverts (/y.js) and DuckDuckGo's own pages
    return href if parts.scheme in ("http", "https") and host else None


class _ResultParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []
        self._field: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a" or self._field is not None:
            return
        values = dict(attrs)
        classes = (values.get("class") or "").split()
        if "result__a" in classes:
            self.results.append({"href": values.get("href") or "", "title": [], "snippet": []})
            self._field = "title"
        elif "result__snippet" in classes and self.results:
            self._field = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._field = None

    def handle_data(self, data: str) -> None:
        if self._field is not None:
            self.results[-1][self._field].append(data)


def parse_results(page: str) -> list[WebResult]:
    parser = _ResultParser()
    parser.feed(page)
    parser.close()
    results = []
    for raw in parser.results:
        url = result_url(raw["href"])
        title = clean_text("".join(raw["title"]))
        if url is None or not title:
            continue
        snippet = clean_text("".join(raw["snippet"]))[:MAX_SNIPPET_CHARS]
        results.append(WebResult(title, url, snippet, "duckduckgo"))
    return results


class DuckDuckGoSearch:
    """Titles, links and snippets from DuckDuckGo's HTML endpoint. Result pages are never fetched.

    This uses httpx rather than the ddgs package: ddgs connects through primp, a Rust HTTP
    client whose connections never pass through Python's socket module or the egress guard.
    """

    name = "duckduckgo"

    def __init__(
        self,
        timeout: float = 10.0,
        proxy: str | None = None,
        client: httpx.Client | None = None,
        region: str = "wt-wt",
    ):
        # trust_env=False: HTTP_PROXY and similar variables must not reroute private searches.
        self.client = client or httpx.Client(proxy=proxy, timeout=timeout, trust_env=False)
        self.proxy = proxy
        self._region = region

    def search(self, query: str, max_results: int) -> list[WebResult]:
        try:
            response = self.client.post(
                SEARCH_URL,
                data={"q": query, "b": "", "l": self._region},
                headers={"User-Agent": USER_AGENT},
            )
        except Exception as exc:
            # Not only httpx.HTTPError: a SOCKS proxy that hangs up mid-handshake surfaces as
            # socksio.exceptions.ProtocolError, which httpx does not wrap.
            raise request_failure("DuckDuckGo", exc) from None
        if response.status_code != 200:
            raise SearchError("web_search_failed", f"DuckDuckGo returned {response.status_code}")
        return parse_results(response.text)[:max_results]
