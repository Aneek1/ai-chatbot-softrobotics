import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict

import httpx

from backend.privacy.egress import egress_block_in
from backend.providers.base import Message, ProviderError


class OllamaModel:
    name = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float,
        client: httpx.Client | None = None,
        repeat_penalty: float = 1.1,
        num_predict: int = 4096,
        repeat_last_n: int = 512,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._client = client or httpx.Client()
        self._repeat_penalty = repeat_penalty
        self._num_predict = num_predict
        self._repeat_last_n = repeat_last_n

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        # With thinking on, qwen3 can spend the whole token budget on hidden reasoning
        # and return an empty answer.
        body = {
            "model": self._model,
            "stream": True,
            "think": False,
            "messages": [asdict(m) for m in messages],
            # temperature/top_p/top_k are left to the model's own Modelfile defaults.
            "options": {
                "repeat_penalty": self._repeat_penalty,
                "num_predict": self._num_predict,
                "repeat_last_n": self._repeat_last_n,
            },
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
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError as exc:
                        detail = f"Ollama sent a line that is not JSON: {line[:200]}"
                        raise ProviderError("provider_error", detail) from exc
                    if "error" in chunk:
                        raise ProviderError("provider_error", f"Ollama error: {chunk['error']}")
                    content = (chunk.get("message") or {}).get("content")
                    if content:
                        yield content
                    if chunk.get("done"):
                        return
        except httpx.ConnectError as exc:
            block = egress_block_in(exc)
            if block is not None:
                raise ProviderError(
                    "egress_blocked", f"The egress guard blocked Ollama at {block.host}"
                ) from exc
            raise ProviderError("ollama_unavailable", f"Ollama is not reachable at {self._base_url}") from exc
        except httpx.TimeoutException as exc:
            detail = f"Ollama did not respond within {self._timeout} s"
            raise ProviderError("provider_timeout", detail) from exc
        except httpx.HTTPError as exc:
            # After the two clauses above: ConnectError and TimeoutException subclass HTTPError.
            # Catches the connection dropping mid-answer (ReadError, RemoteProtocolError).
            raise ProviderError("provider_error", f"Ollama request failed: {exc}") from exc

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
