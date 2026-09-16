from types import SimpleNamespace

import httpx
import pytest

from backend.privacy.egress import EgressBlocked
from backend.providers.base import Message, ProviderError
from backend.providers.gemini import GeminiModel


class FakeModels:
    def __init__(self, texts=(), error=None):
        self._texts = texts
        self._error = error
        self.calls = []

    def generate_content_stream(self, *, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self._error:
            raise self._error
        for text in self._texts:
            yield SimpleNamespace(text=text)


def gemini(models):
    return GeminiModel(api_key="unused", model="test-model", client=SimpleNamespace(models=models))


def test_streams_text_and_maps_roles():
    models = FakeModels(texts=["Halo", None, " dunia"])
    messages = [
        Message("system", "Answer in Indonesian."),
        Message("user", "hi"),
        Message("assistant", "Halo"),
        Message("user", "lagi"),
    ]
    assert list(gemini(models).stream(messages)) == ["Halo", " dunia"]

    call = models.calls[0]
    assert call["model"] == "test-model"
    assert call["config"] == {"system_instruction": "Answer in Indonesian."}
    assert [c["role"] for c in call["contents"]] == ["user", "model", "user"]
    assert call["contents"][0]["parts"] == [{"text": "hi"}]


def test_sdk_errors_become_provider_errors():
    models = FakeModels(error=RuntimeError("quota exceeded"))
    with pytest.raises(ProviderError) as error:
        list(gemini(models).stream([Message("user", "hi")]))
    assert error.value.code == "provider_error"
    assert "quota exceeded" in str(error.value)


def test_timeout_is_provider_timeout():
    models = FakeModels(error=httpx.ReadTimeout("timed out"))
    with pytest.raises(ProviderError) as error:
        list(gemini(models).stream([Message("user", "hi")]))
    assert error.value.code == "provider_timeout"


def test_real_client_gets_a_timeout_in_milliseconds(monkeypatch):
    from google import genai

    created = {}

    def fake_client(**kwargs):
        created.update(kwargs)
        return SimpleNamespace(models=FakeModels())

    monkeypatch.setattr(genai, "Client", fake_client)
    GeminiModel(api_key="test-key", model="test-model", timeout=2.5)
    assert created["api_key"] == "test-key"
    assert created["http_options"].timeout == 2500


def test_blocked_connection_is_egress_blocked():
    error_from_sdk = httpx.ConnectError("blocked")
    error_from_sdk.__cause__ = EgressBlocked("generativelanguage.googleapis.com", 443)
    with pytest.raises(ProviderError) as error:
        list(gemini(FakeModels(error=error_from_sdk)).stream([Message("user", "hi")]))
    assert error.value.code == "egress_blocked"
    assert "generativelanguage.googleapis.com" in str(error.value)
