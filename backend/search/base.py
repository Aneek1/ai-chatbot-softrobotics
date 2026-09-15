import re
from dataclasses import dataclass
from typing import Protocol

import httpx

from backend.privacy.egress import egress_block_in

_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    snippet: str
    engine: str


class SearchError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class WebSearch(Protocol):
    name: str

    def search(self, query: str, max_results: int) -> list[WebResult]: ...


def clean_text(text: str) -> str:
    return _SPACE.sub(" ", text).strip()


def request_failure(engine: str, exc: Exception) -> SearchError:
    """Describe a failed request without the exception text, which can contain the URL and its API key."""
    block = egress_block_in(exc)
    if block is not None:
        return SearchError("egress_blocked", f"The egress guard blocked {engine} ({block.host})")
    if isinstance(exc, httpx.TimeoutException):
        return SearchError("web_search_failed", f"{engine} did not respond in time")
    return SearchError("web_search_failed", f"{engine} request failed ({type(exc).__name__})")
