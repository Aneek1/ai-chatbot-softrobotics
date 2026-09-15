from collections.abc import Iterator, Sequence
from typing import Any

import httpx

from backend.providers.base import Message, ProviderError


class GeminiModel:
    name = "gemini"

    def __init__(self, api_key: str, model: str, timeout: float = 120.0, client: Any = None):
        if client is None:
            from google import genai

            # The SDK sets no timeout by default, so a stalled request would hang forever.
            # HttpOptions.timeout is in milliseconds.
            options = genai.types.HttpOptions(timeout=int(timeout * 1000))
            client = genai.Client(api_key=api_key, http_options=options)
        self._client = client
        self._model = model
        self._timeout = timeout

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        contents = [
            {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
            for m in messages
            if m.role != "system"
        ]
        config = {"system_instruction": system} if system else None
        try:
            for chunk in self._client.models.generate_content_stream(
                model=self._model, contents=contents, config=config
            ):
                if chunk.text:
                    yield chunk.text
        except ProviderError:
            raise
        except httpx.TimeoutException as exc:
            detail = f"Gemini did not respond within {self._timeout} s"
            raise ProviderError("provider_timeout", detail) from exc
        except Exception as exc:  # the SDK raises several unrelated error types
            raise ProviderError("provider_error", f"Gemini request failed: {exc}") from exc
