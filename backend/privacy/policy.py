import ipaddress
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

from backend.app.config import Settings

LOCAL_NAMES = frozenset({"localhost"})
GEMINI_HOST = "generativelanguage.googleapis.com"
GOOGLE_SEARCH_HOST = "www.googleapis.com"
DUCKDUCKGO_HOST = "html.duckduckgo.com"


def normalize_host(host: str | bytes) -> str:
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    host = host.strip().lower().rstrip(".")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host.split("%", 1)[0]  # drop an IPv6 zone index such as %12


def is_loopback(host: str) -> bool:
    if host in LOCAL_NAMES:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return address.is_loopback or bool(mapped and mapped.is_loopback)


@dataclass(frozen=True)
class EgressPolicy:
    normal: frozenset[str]
    private: frozenset[str]
    local: frozenset[str] = frozenset()  # allowed in both modes, like loopback

    def allows(self, host: str | bytes, private: bool) -> bool:
        name = normalize_host(host)
        if is_loopback(name) or name in self.local:
            return True
        return name in (self.private if private else self.normal)


def host_and_port(url: str) -> tuple[str, int]:
    parts = urlsplit(url)
    default = 443 if parts.scheme == "https" else 80
    return normalize_host(parts.hostname or ""), parts.port or default


def build_policy(settings: Settings) -> EgressPolicy:
    normal = {GEMINI_HOST, GOOGLE_SEARCH_HOST}
    if settings.web_search == "duckduckgo":
        normal.add(DUCKDUCKGO_HOST)
    if settings.private_proxy:
        private = {host_and_port(settings.private_proxy)[0]}
    else:
        private = {DUCKDUCKGO_HOST}
    # OLLAMA_URL is chosen by whoever runs the app, and answers must reach it in both modes.
    ollama_host, _ = host_and_port(settings.ollama_url)
    return EgressPolicy(normal=frozenset(normal), private=frozenset(private), local=frozenset({ollama_host}))


def build_labels(settings: Settings) -> dict[str, str]:
    ollama_host, ollama_port = host_and_port(settings.ollama_url)
    labels = {
        GEMINI_HOST: "gemini",
        GOOGLE_SEARCH_HOST: "google_search",
        DUCKDUCKGO_HOST: "duckduckgo",
        f"{ollama_host}:{ollama_port}": "ollama",
    }
    if settings.private_proxy:
        proxy_host, proxy_port = host_and_port(settings.private_proxy)
        labels[f"{proxy_host}:{proxy_port}"] = "private_proxy"
    return labels


def component_for(labels: Mapping[str, str], host: str, port: int | None) -> str:
    return labels.get(f"{host}:{port}") or labels.get(host) or "other"
