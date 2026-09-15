import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict

import httpx

from backend.providers.base import Message, ProviderError


class OllamaModel:
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float, client: httpx.Client | None = None):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._client = client or httpx.Client()

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        # With thinking on, qwen3 can spend the whole token budget on hidden reasoning
        # and return an empty answer.
        body = {
            "model": self._model,
            "stream": True,
            "think": False,
            "messages": [asdict(m) for m in messages],
        }
        try:
            with self._client.stream(
                "POST", f"{self._base_url}/api/chat", json=body, timeout=self._timeout
            ) as response:
                if response.status_code != 200:
                    response.read()
                    raise ProviderError(
                        "provider_error", f"Ollama returned {response.status_code}: {response.text[:200]}"
                    )
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise ProviderError("provider_error", f"Ollama error: {chunk['error']}")
                    content = chunk.get("message", {}).get("content")
                    if content:
                        yield content
                    if chunk.get("done"):
                        return
        except httpx.ConnectError as exc:
            raise ProviderError("ollama_unavailable", f"Ollama is not reachable at {self._base_url}") from exc
        except httpx.TimeoutException as exc:
            detail = f"Ollama did not respond within {self._timeout} s"
            raise ProviderError("provider_timeout", detail) from exc

    def reachable(self) -> bool:
        try:
            return self._client.get(f"{self._base_url}/api/tags", timeout=3.0).status_code == 200
        except httpx.HTTPError:
            return False

    def warm(self) -> bool:
        """An empty prompt makes Ollama load the model into memory without generating."""
        try:
            response = self._client.post(
                f"{self._base_url}/api/generate",
                json={"model": self._model, "prompt": "", "keep_alive": "10m"},
                timeout=self._timeout,
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False
