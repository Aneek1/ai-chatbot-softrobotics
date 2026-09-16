import json

import httpx
from fastapi.testclient import TestClient

from backend.app.main import create_app
from tests.api_fakes import fake_services, parse_sse, serve


def logged_services():
    services = fake_services()
    services.egress_log.record("localhost", 11434, "allowed", "ollama")
    services.egress_log.record("example.com", 443, "blocked", "other")
    return services


def test_snapshot_lists_logged_connections():
    with TestClient(create_app(logged_services())) as client:
        response = client.get("/api/egress", params={"follow": "false"})
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [e for e, _ in events] == ["connection", "connection"]
    assert [(d["host"], d["port"], d["verdict"], d["component"]) for _, d in events] == [
        ("localhost", 11434, "allowed", "ollama"),
        ("example.com", 443, "blocked", "other"),
    ]
    assert set(events[0][1]) == {"time", "host", "port", "verdict", "component"}
    assert "id: 1" in response.text


def test_last_event_id_skips_entries_the_client_has_seen():
    with TestClient(create_app(logged_services())) as client:
        response = client.get("/api/egress", params={"follow": "false"}, headers={"Last-Event-ID": "1"})
    assert [d["host"] for _, d in parse_sse(response.text)] == ["example.com"]


def test_unreadable_last_event_id_starts_from_the_oldest_entry():
    with TestClient(create_app(logged_services())) as client:
        response = client.get("/api/egress", params={"follow": "false"}, headers={"Last-Event-ID": "abc"})
    assert len(parse_sse(response.text)) == 2


def test_live_stream_delivers_connections_logged_after_it_opened():
    services = fake_services()
    services.egress_log.record("localhost", 11434, "allowed", "ollama")
    hosts = []
    with serve(create_app(services)) as base_url, httpx.Client(timeout=10) as client:
        with client.stream("GET", f"{base_url}/api/egress") as response:
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                hosts.append(json.loads(line.removeprefix("data: "))["host"])
                if len(hosts) == 1:
                    services.egress_log.record("html.duckduckgo.com", 443, "allowed", "duckduckgo")
                else:
                    break
    assert hosts == ["localhost", "html.duckduckgo.com"]
