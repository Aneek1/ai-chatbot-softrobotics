from collections.abc import Iterator, Sequence

from backend.pipeline.langid import Candidate
from backend.providers.base import Message, ProviderError


def word_count(text: str) -> int:
    return len(text.split())


class FakeScorer:
    """Returns scripted candidates; records the texts it was asked about."""

    def __init__(self, candidates: list[tuple[str, float]], *, fail_if_called: bool = False):
        self._candidates = [Candidate(code, p) for code, p in candidates]
        self._fail_if_called = fail_if_called
        self.seen: list[str] = []

    def top(self, text: str, k: int) -> list[Candidate]:
        if self._fail_if_called:
            raise AssertionError("scorer should not have been called")
        assert "\n" not in text, "detectors must receive normalized text"
        self.seen.append(text)
        return self._candidates[:k]


class FakeEmbedder:
    """Bag-of-keywords vectors, plus a constant so no vector is all zeros."""

    KEYWORDS = ("silicone", "printing", "market")
    dim = len(KEYWORDS) + 1

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        return [float(lowered.count(k)) for k in self.KEYWORDS] + [0.01]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class FakeModel:
    name = "fake"

    def __init__(self, pieces: Sequence[str] = ("Hello",), error: ProviderError | None = None):
        self._pieces = pieces
        self._error = error
        self.calls: list[list[Message]] = []

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        self.calls.append(list(messages))
        if self._error is not None:
            raise self._error
        yield from self._pieces
