import socket

import pytest

import backend.app.services as services_module
from backend.app.config import Settings
from backend.app.services import build_privacy, build_services
from backend.privacy.egress import EgressBlocked
from backend.search.duckduckgo import DuckDuckGoSearch
from backend.search.google import GoogleSearch
from backend.store.history import Exchange
from tests.fakes import FakeEmbedder, FakeScorer


class FakeIndex:
    @staticmethod
    def open(path, embedder):
        return FakeIndex()

    def close(self):
        pass


def patch_models(monkeypatch) -> list:
    loaded = []

    def fake_load(models_dir, name, embedder=None):
        loaded.append((name, embedder))
        return FakeScorer([("zsm_Latn", 0.9), ("ind_Latn", 0.1)])

    monkeypatch.setattr(
        services_module, "FastTextScorer", lambda path: FakeScorer([("ind_Latn", 0.7), ("zsm_Latn", 0.3)])
    )
    monkeypatch.setattr(services_module, "E5Embedder", lambda path: FakeEmbedder())
    monkeypatch.setattr(services_module, "ChunkIndex", FakeIndex)
    monkeypatch.setattr(services_module, "load_specialist", fake_load)
    return loaded


@pytest.fixture
def settings(tmp_path):
    def make(**overrides) -> Settings:
        values = {
            "google_api_key": None,
            "gemini_model": None,
            "google_search_key": None,
            "google_search_engine_id": None,
            "private_mode": False,
            "private_proxy": None,
            "web_search": "google",
            "history_db": tmp_path / "history.db",
        }
        return Settings(_env_file=None, **(values | overrides))

    return make


@pytest.fixture
def build(monkeypatch):
    patch_models(monkeypatch)
    built = []

    def make(settings):
        services = build_services(settings)
        built.append(services)
        return services

    yield make
    for services in built:
        services.close()


def test_no_specialist_by_default(monkeypatch, settings):
    loaded = patch_models(monkeypatch)
    services = build_services(settings(langid_specialist=None))
    services.close()
    assert loaded == []
    assert services.detector_name == "glotlid-q.ftz"
    assert services.chat.detector.detect("Bagaimana cara membuat").stage == "general"


def test_empty_setting_means_no_specialist(monkeypatch, settings):
    loaded = patch_models(monkeypatch)
    build_services(settings(langid_specialist="")).close()
    assert loaded == []


def test_configured_specialist_becomes_stage_two(monkeypatch, settings):
    loaded = patch_models(monkeypatch)
    services = build_services(settings(langid_specialist="lid-specialist-e5-head.onnx"))
    services.close()
    assert loaded[0][0] == "lid-specialist-e5-head.onnx"
    assert isinstance(loaded[0][1], FakeEmbedder)
    assert services.detector_name == "glotlid-q.ftz + lid-specialist-e5-head.onnx"
    result = services.chat.detector.detect("Bagaimana cara membuat")
    assert result.chosen == "zsm_Latn"
    assert result.stage == "specialist"


def test_guard_is_built_but_left_for_the_app_to_install(build, settings):
    services = build(settings())
    assert services.guard is not None
    assert services.guard.installed is False
    assert services.guard.log is services.egress_log


def test_guard_follows_the_mode_switch(monkeypatch, settings):
    looked_up = []
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, *args, **kwargs: looked_up.append(host) or [])
    privacy = build_privacy(settings())
    privacy.guard.install()
    try:
        with pytest.raises(EgressBlocked):
            socket.getaddrinfo("html.duckduckgo.com", 443)
        privacy.mode.set_private(True, timeout=1)
        socket.getaddrinfo("html.duckduckgo.com", 443)
        with pytest.raises(EgressBlocked):
            socket.getaddrinfo("generativelanguage.googleapis.com", 443)
    finally:
        privacy.guard.uninstall()
    assert looked_up == ["html.duckduckgo.com"]


def test_given_privacy_objects_are_used(monkeypatch, settings):
    patch_models(monkeypatch)
    privacy = build_privacy(settings())
    services = build_services(settings(), privacy)
    services.close()
    assert services.mode is privacy.mode
    assert services.guard is privacy.guard
    assert services.egress_log is privacy.egress_log


def test_starts_in_private_mode_when_configured(build, settings):
    assert build(settings(private_mode=True)).mode.private is True


def test_normal_search_is_google_when_its_key_and_engine_are_set(build, settings):
    services = build(settings(google_search_key="key", google_search_engine_id="engine"))
    assert isinstance(services.chat.normal_search, GoogleSearch)


def test_normal_search_is_off_without_google_credentials(build, settings):
    assert build(settings()).chat.normal_search is None


def test_normal_search_can_use_duckduckgo(build, settings):
    services = build(
        settings(web_search="duckduckgo", google_search_key="key", google_search_engine_id="engine")
    )
    assert isinstance(services.chat.normal_search, DuckDuckGoSearch)


def test_normal_search_can_be_turned_off(build, settings):
    services = build(settings(web_search="off", google_search_key="key", google_search_engine_id="engine"))
    assert services.chat.normal_search is None


def test_private_search_is_duckduckgo_through_the_configured_proxy(build, settings):
    services = build(settings(private_proxy="socks5h://127.0.0.1:9050", search_max_results=3))
    assert isinstance(services.chat.private_search, DuckDuckGoSearch)
    assert services.chat.private_search.proxy == "socks5h://127.0.0.1:9050"
    assert services.chat.max_web_results == 3


def test_normal_history_is_a_file_and_private_history_is_separate(build, settings, tmp_path):
    services = build(settings())
    assert (tmp_path / "history.db").exists()
    assert services.private_history is not services.history


def test_leaving_private_mode_clears_private_chats(build, settings):
    services = build(settings(private_mode=True))
    exchange = Exchange("q", {}, "a", "eng_Latn", [], [], {"valid": [], "removed": []})
    services.private_history.add_exchange("p1", exchange)
    services.mode.set_private(False, timeout=1)
    assert services.private_history.list_chats() == []
