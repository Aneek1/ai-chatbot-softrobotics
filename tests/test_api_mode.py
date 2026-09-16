from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.privacy.mode import ModeState
from backend.store.history import Exchange
from tests.api_fakes import fake_services

EXCHANGE = Exchange("q", {}, "a", "eng_Latn", [], [], {"valid": [], "removed": []})


class FakeOllama:
    def __init__(self, reachable: bool):
        self._reachable = reachable

    def reachable(self) -> bool:
        return self._reachable


def test_get_mode_reports_private_proxy_and_ollama():
    with TestClient(create_app(fake_services(ollama=FakeOllama(True)))) as client:
        body = client.get("/api/mode").json()
    assert body == {"private": False, "proxy_configured": False, "ollama_reachable": True}


def test_proxy_is_reported_when_configured():
    settings = Settings(_env_file=None, private_proxy="socks5h://127.0.0.1:9050")
    with TestClient(create_app(fake_services(settings=settings))) as client:
        assert client.get("/api/mode").json()["proxy_configured"] is True


def test_put_switches_the_mode_and_returns_the_new_state():
    services = fake_services()
    with TestClient(create_app(services)) as client:
        response = client.put("/api/mode", json={"private": True})
        assert services.mode.private is True
    assert response.status_code == 200
    assert response.json() == {"private": True, "proxy_configured": False, "ollama_reachable": False}


def test_switch_happens_even_when_ollama_is_unreachable():
    services = fake_services(ollama=FakeOllama(False))
    with TestClient(create_app(services)) as client:
        response = client.put("/api/mode", json={"private": True})
        assert services.mode.private is True
    assert response.json()["ollama_reachable"] is False


def test_switching_back_clears_private_chats():
    services = fake_services(mode=ModeState(private=True))
    with TestClient(create_app(services)) as client:
        services.private_history.add_exchange("p1", EXCHANGE)
        client.put("/api/mode", json={"private": False})
        assert services.private_history.list_chats() == []


def test_switch_fails_with_409_while_an_answer_is_still_running():
    settings = Settings(_env_file=None, mode_switch_timeout=0.05)
    services = fake_services(settings=settings)
    with TestClient(create_app(services)) as client:
        services.mode.begin_answer()
        try:
            response = client.put("/api/mode", json={"private": True})
        finally:
            services.mode.end_answer()
        assert services.mode.private is False
    assert response.status_code == 409
    assert "mode unchanged" in response.json()["detail"]


def test_mode_must_be_a_real_boolean():
    with TestClient(create_app(fake_services())) as client:
        assert client.put("/api/mode", json={"private": "yes"}).status_code == 422
