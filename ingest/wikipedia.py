"""Wikipedia lead sections in the supported languages, with the revision id of each page."""

import time
from collections.abc import Callable, Sequence
from urllib.parse import quote

import httpx

from ingest.documents import Document, IngestError, make_document, today

LICENCE = "CC BY-SA 4.0"
USER_AGENT = "softrobotics-assistant/0.1 (knowledge-base build)"
# Wiki codes for the eight supported languages. The Chinese Wikipedia is one wiki with script
# variants, so zho_Hant has no separate wiki to fetch; the datasheet records that gap.
WIKI_LANGUAGES = {
    "en": "eng_Latn",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ta": "tam_Taml",
    "hi": "hin_Deva",
}


def api_url(wiki_lang: str) -> str:
    return f"https://{wiki_lang}.wikipedia.org/w/api.php"


def page_url(wiki_lang: str, title: str) -> str:
    return f"https://{wiki_lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"


def _page(payload: dict) -> dict | None:
    pages = (payload.get("query") or {}).get("pages") or []
    if not pages:
        return None
    page = pages[0]
    if page.get("missing") or not page.get("extract"):
        return None
    return page


def parse_page(payload: dict, wiki_lang: str, retrieved_at: str) -> Document | None:
    page = _page(payload)
    if page is None:
        return None
    revisions = page.get("revisions") or [{}]
    return make_document(
        id=f"wikipedia:{wiki_lang}:{page['pageid']}",
        text=page["extract"],
        language=WIKI_LANGUAGES[wiki_lang],
        source="wikipedia",
        title=page["title"],
        url=page_url(wiki_lang, page["title"]),
        licence=LICENCE,
        retrieved_at=retrieved_at,
        revision=str(revisions[0].get("revid", "")),
    )


def parse_langlinks(payload: dict) -> dict[str, str]:
    page = _page(payload)
    if page is None:
        return {}
    return {
        link["lang"]: link["title"]
        for link in page.get("langlinks") or []
        if link["lang"] in WIKI_LANGUAGES and link["lang"] != "en"
    }


def retry_delay(response: httpx.Response, default: float, cap: float = 60.0) -> float:
    """Seconds to wait after a 429: the Retry-After header when the API sends a usable one."""
    try:
        seconds = float(response.headers.get("Retry-After", ""))
    except ValueError:
        seconds = default
    return max(0.0, min(seconds, cap))


class WikipediaClient:
    def __init__(
        self,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
        pause: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
        retries: int = 2,
    ):
        self._client = client or httpx.Client(timeout=timeout)
        self._pause = pause
        self._sleep = sleep
        self._retries = retries
        self._requested = False

    def _get(self, wiki_lang: str, params: dict[str, str]) -> dict:
        if self._requested:
            self._sleep(self._pause)
        self._requested = True
        attempts = self._retries + 1
        for attempt in range(attempts):
            try:
                response = self._client.get(
                    api_url(wiki_lang), params=params, headers={"User-Agent": USER_AGENT}
                )
            except httpx.HTTPError as exc:
                raise IngestError(f"Wikipedia request failed ({type(exc).__name__})") from exc
            if response.status_code == 429 and attempt + 1 < attempts:
                # The API asks callers to back off and says for how long. One 429 must not throw
                # away every page fetched so far, so wait the delay out and ask again.
                self._sleep(retry_delay(response, self._pause))
                continue
            if response.status_code != 200:
                raise IngestError(f"{wiki_lang}.wikipedia.org returned {response.status_code}")
            return response.json()
        raise IngestError(f"{wiki_lang}.wikipedia.org returned 429")  # pragma: no cover

    def page(
        self, wiki_lang: str, title: str, retrieved_at: str | None = None
    ) -> tuple[Document | None, dict[str, str]]:
        if wiki_lang not in WIKI_LANGUAGES:
            raise IngestError(f"unsupported Wikipedia language '{wiki_lang}'")
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "extracts|langlinks|revisions",
            "explaintext": "1",
            "exintro": "1",
            "redirects": "1",
            "lllimit": "500",
            "rvprop": "ids|timestamp",
            "titles": title,
        }
        if wiki_lang == "zh":
            params["variant"] = "zh-hans"
        payload = self._get(wiki_lang, params)
        return parse_page(payload, wiki_lang, retrieved_at or today()), parse_langlinks(payload)

    def articles(self, english_titles: Sequence[str], retrieved_at: str | None = None) -> list[Document]:
        """Each English article, then the same article in the supported languages that have one."""
        when = retrieved_at or today()
        documents: list[Document] = []
        for title in english_titles:
            english, links = self.page("en", title, when)
            if english is None:
                continue
            documents.append(english)
            for wiki_lang, other_title in sorted(links.items()):
                translated, _ = self.page(wiki_lang, other_title, when)
                if translated is not None:
                    documents.append(translated)
        return documents
