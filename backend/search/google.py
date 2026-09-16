import httpx

from backend.search.base import SearchError, WebResult, clean_text, request_failure

SEARCH_URL = "https://www.googleapis.com/customsearch/v1"
MAX_PER_REQUEST = 10  # the Custom Search JSON API rejects larger `num` values


class GoogleSearch:
    """Titles, links and snippets from the Google Custom Search JSON API. Result pages are never fetched."""

    name = "google"

    def __init__(
        self, api_key: str, engine_id: str, timeout: float = 10.0, client: httpx.Client | None = None
    ):
        self._key = api_key
        self._engine_id = engine_id
        self.client = client or httpx.Client(timeout=timeout, trust_env=False)

    def search(self, query: str, max_results: int) -> list[WebResult]:
        params = {
            "key": self._key,
            "cx": self._engine_id,
            "q": query,
            "num": min(max_results, MAX_PER_REQUEST),
        }
        try:
            response = self.client.get(SEARCH_URL, params=params)
        except httpx.HTTPError as exc:
            # `from None`: the original exception can carry the request URL, which holds the key.
            raise request_failure("Google search", exc) from None
        if response.status_code != 200:
            detail = self._error_message(response)
            raise SearchError("web_search_failed", f"Google search returned {response.status_code}: {detail}")
        try:
            data = response.json()
        except ValueError:
            raise SearchError("web_search_failed", "Google search sent a response that is not JSON") from None
        items = data.get("items", []) if isinstance(data, dict) else []
        results = [
            WebResult(clean_text(item["title"]), item["link"], clean_text(item.get("snippet", "")), "google")
            for item in items
            if isinstance(item, dict) and item.get("title") and item.get("link")
        ]
        return results[:max_results]

    def _error_message(self, response: httpx.Response) -> str:
        try:
            message = str(response.json()["error"]["message"])[:200]
        except (ValueError, KeyError, TypeError):
            return "no error message"
        return message.replace(self._key, "***")
