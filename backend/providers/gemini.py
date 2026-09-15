from collections.abc import Iterator, Sequence
from typing import Any

from backend.providers.base import Message, ProviderError


class GeminiModel:
    name = "gemini"

    def __init__(self, api_key: str, model: str, client: Any = None):
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key)
        self._client = client
        self._model = model

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
        except Exception as exc:  # the SDK raises several unrelated error types
            raise ProviderError("provider_error", f"Gemini request failed: {exc}") from exc
