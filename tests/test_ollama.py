import json

import httpx
import pytest

from backend.providers.base import Message, ProviderError
from backend.providers.ollama import OllamaModel


def model_with(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaModel("http://ollama.test", "qwen3:8b", timeout=5.0, client=client)


def ndjson(*objects):
    return "\n".join(json.dumps(o) for o in objects).encode()


def test_streams_content_and_disables_thinking():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=ndjson(
                {"message": {"role": "assistant", "content": "Halo"}, "done": False},
                {"message": {"role": "assistant", "content": " dunia"}, "done": False},
                {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"},
            ),
        )

    pieces = list(model_with(handler).stream([Message("user", "hi")]))
    assert pieces == ["Halo", " dunia"]
    assert seen["body"]["think"] is False
    assert seen["body"]["stream"] is True
    assert seen["body"]["messages"] == [{"role": "user", "content": "hi"}]


def test_connection_refused_is_ollama_unavailable():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderError) as error:
        list(model_with(handler).stream([Message("user", "hi")]))
    assert error.value.code == "ollama_unavailable"


def test_timeout_is_provider_timeout():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ProviderError) as error:
        list(model_with(handler).stream([Message("user", "hi")]))
    assert error.value.code == "provider_timeout"


def test_error_status_is_provider_error():
    def handler(request):
        return httpx.Response(404, json={"error": "model not found"})

    with pytest.raises(ProviderError) as error:
        list(model_with(handler).stream([Message("user", "hi")]))
    assert error.value.code == "provider_error"
    assert "404" in str(error.value)


def test_error_line_in_stream_is_provider_error():
    def handler(request):
        return httpx.Response(200, content=ndjson({"error": "out of memory"}))

    with pytest.raises(ProviderError, match="out of memory"):
        list(model_with(handler).stream([Message("user", "hi")]))


class BrokenStream(httpx.SyncByteStream):
    """Sends one NDJSON line, then drops the connection, as when Ollama crashes mid-answer."""

    def __iter__(self):
        yield ndjson({"message": {"role": "assistant", "content": "Halo"}, "done": False}) + b"\n"
        raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")


def test_connection_lost_mid_stream_is_provider_error():
    def handler(request):
        return httpx.Response(200, stream=BrokenStream())

    pieces = []
    with pytest.raises(ProviderError) as error:
        for piece in model_with(handler).stream([Message("user", "hi")]):
            pieces.append(piece)
    assert pieces == ["Halo"]
    assert error.value.code == "provider_error"


def test_non_json_line_is_provider_error():
    def handler(request):
        return httpx.Response(200, content=b"<html>proxy</html>")

    with pytest.raises(ProviderError) as error:
        list(model_with(handler).stream([Message("user", "hi")]))
    assert error.value.code == "provider_error"
    assert "not JSON" in str(error.value)


def test_null_message_is_skipped():
    def handler(request):
        return httpx.Response(200, content=ndjson({"message": None, "done": True}))

    assert list(model_with(handler).stream([Message("user", "hi")])) == []


def test_reachable_and_warm():
    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/api/generate":
            assert json.loads(request.content)["prompt"] == ""
            return httpx.Response(200, json={"done": True})
        return httpx.Response(404)

    model = model_with(handler)
    assert model.reachable() is True
    assert model.warm() is True


def test_unreachable():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    assert model_with(handler).reachable() is False
