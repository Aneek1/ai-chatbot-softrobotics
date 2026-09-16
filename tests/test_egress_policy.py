import pytest

from backend.app.config import Settings
from backend.privacy.policy import (
    DUCKDUCKGO_HOST,
    GEMINI_HOST,
    GOOGLE_SEARCH_HOST,
    build_labels,
    build_policy,
    component_for,
    is_loopback,
    normalize_host,
)


def settings(**overrides) -> Settings:
    values = {
        "google_api_key": None,
        "gemini_model": None,
        "private_proxy": None,
        "web_search": "google",
        "ollama_url": "http://localhost:11434",
    }
    return Settings(_env_file=None, **(values | overrides))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("WWW.GoogleAPIs.com.", "www.googleapis.com"),
        (b"localhost", "localhost"),
        ("[::1]", "::1"),
        ("fe80::1%12", "fe80::1"),
    ],
)
def test_normalize_host(raw, expected):
    assert normalize_host(raw) == expected


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "127.5.0.1", "::1", "::ffff:127.0.0.1"])
def test_loopback_hosts(host):
    assert is_loopback(host)


@pytest.mark.parametrize("host", ["192.168.1.10", "example.com", "localhost.example.com", "0.0.0.0"])
def test_hosts_that_are_not_loopback(host):
    assert not is_loopback(host)


def test_normal_mode_allows_the_google_hosts_and_local_addresses():
    policy = build_policy(settings())
    assert policy.allows(GEMINI_HOST, private=False)
    assert policy.allows(GOOGLE_SEARCH_HOST, private=False)
    assert policy.allows("::1", private=False)
    assert not policy.allows(DUCKDUCKGO_HOST, private=False)
    assert not policy.allows("example.com", private=False)


def test_private_mode_allows_duckduckgo_and_local_addresses_only():
    policy = build_policy(settings())
    assert policy.allows(DUCKDUCKGO_HOST, private=True)
    for host in ("localhost", "127.0.0.1", "::1"):
        assert policy.allows(host, private=True)
    assert not policy.allows(GEMINI_HOST, private=True)
    assert not policy.allows(GOOGLE_SEARCH_HOST, private=True)


def test_proxy_is_the_only_outside_host_in_private_mode():
    policy = build_policy(settings(private_proxy="socks5h://tor.lan:9050"))
    assert policy.allows("tor.lan", private=True)
    assert not policy.allows(DUCKDUCKGO_HOST, private=True)
    assert not policy.allows("tor.lan", private=False)


def test_duckduckgo_setting_allows_it_in_normal_mode():
    policy = build_policy(settings(web_search="duckduckgo"))
    assert policy.allows(DUCKDUCKGO_HOST, private=False)


def test_configured_ollama_host_is_allowed_in_both_modes():
    policy = build_policy(settings(ollama_url="http://ollama:11434"))
    assert policy.allows("ollama", private=True)
    assert policy.allows("ollama", private=False)


def test_labels_name_each_known_host():
    labels = build_labels(settings(private_proxy="socks5h://tor.lan:9050"))
    assert labels == {
        GEMINI_HOST: "gemini",
        GOOGLE_SEARCH_HOST: "google_search",
        DUCKDUCKGO_HOST: "duckduckgo",
        "localhost:11434": "ollama",
        "tor.lan:9050": "private_proxy",
    }


def test_component_prefers_host_and_port_then_host():
    labels = {"localhost:11434": "ollama", GEMINI_HOST: "gemini"}
    assert component_for(labels, "localhost", 11434) == "ollama"
    assert component_for(labels, "localhost", 5173) == "other"
    assert component_for(labels, GEMINI_HOST, 443) == "gemini"
