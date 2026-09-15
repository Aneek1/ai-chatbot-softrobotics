import asyncio
import socket
from asyncio import proactor_events
from types import SimpleNamespace

import httpx
import pytest

from backend.privacy.egress import EgressBlocked, EgressGuard, egress_block_in
from backend.privacy.egress_log import EgressLog
from backend.privacy.policy import EgressPolicy

POLICY = EgressPolicy(
    normal=frozenset({"generativelanguage.googleapis.com"}),
    private=frozenset({"html.duckduckgo.com"}),
)
TEST_NET_ADDRESS = "192.0.2.10"  # RFC 5737 documentation range, never routed
GEMINI = "generativelanguage.googleapis.com"


class FakeResolver:
    def __init__(self):
        self.calls = []

    def __call__(self, host, port, family=0, type=0, proto=0, flags=0):
        self.calls.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (TEST_NET_ADDRESS, port))]


class FakeConnect:
    def __init__(self):
        self.calls = []

    def __call__(self, sock, address):
        self.calls.append(address)

    def connect_ex(self, sock, address):
        self.calls.append(address)
        return 0


@pytest.fixture
def fake_network(monkeypatch):
    """Stand-ins for DNS and connect, so what the guard lets through never reaches the network."""
    resolver = FakeResolver()
    connect = FakeConnect()
    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect.connect_ex)
    return SimpleNamespace(resolver=resolver, connect=connect)


def make_guard(policy=POLICY, private=False):
    mode = SimpleNamespace(private=private)
    log = EgressLog()
    guard = EgressGuard(policy, lambda: mode.private, log, labels={"html.duckduckgo.com": "duckduckgo"})
    return SimpleNamespace(guard=guard, mode=mode, log=log)


@pytest.fixture
def guarded():
    made = make_guard()
    made.guard.install()
    yield made
    made.guard.uninstall()


def logged(guarded):
    return [(e.host, e.port, e.verdict, e.component) for e in guarded.log.entries()]


def test_blocked_name_raises_before_any_dns_query(fake_network, guarded):
    with pytest.raises(EgressBlocked) as error:
        socket.getaddrinfo("example.com", 443)
    assert fake_network.resolver.calls == []
    assert (error.value.host, error.value.port) == ("example.com", 443)
    assert logged(guarded) == [("example.com", 443, "blocked", "other")]


def test_allowed_name_resolves_and_its_address_connects(fake_network, guarded):
    socket.getaddrinfo(GEMINI, 443)
    with socket.socket() as sock:
        sock.connect((TEST_NET_ADDRESS, 443))
    assert fake_network.resolver.calls == [GEMINI]
    assert fake_network.connect.calls == [(TEST_NET_ADDRESS, 443)]
    assert logged(guarded) == [(GEMINI, 443, "allowed", "other")]


def test_address_resolved_in_normal_mode_is_blocked_after_switching_to_private(fake_network, guarded):
    socket.getaddrinfo(GEMINI, 443)
    guarded.mode.private = True
    with socket.socket() as sock, pytest.raises(EgressBlocked):
        sock.connect((TEST_NET_ADDRESS, 443))
    assert fake_network.connect.calls == []


def test_address_that_no_allowed_lookup_returned_is_blocked(fake_network, guarded):
    with socket.socket() as sock, pytest.raises(EgressBlocked):
        sock.connect(("203.0.113.5", 443))
    assert fake_network.connect.calls == []


def test_name_passed_straight_to_connect_is_checked_before_resolution(fake_network, guarded):
    with socket.socket() as sock, pytest.raises(EgressBlocked):
        sock.connect(("example.com", 80))
    assert fake_network.connect.calls == []
    assert fake_network.resolver.calls == []


def test_connect_ex_is_guarded(fake_network, guarded):
    with socket.socket() as sock, pytest.raises(EgressBlocked):
        sock.connect_ex(("203.0.113.5", 443))
    assert fake_network.connect.calls == []


def test_private_mode_allows_duckduckgo_and_blocks_gemini(fake_network, guarded):
    guarded.mode.private = True
    socket.getaddrinfo("html.duckduckgo.com", 443)
    with socket.socket() as sock:
        sock.connect((TEST_NET_ADDRESS, 443))
    with pytest.raises(EgressBlocked):
        socket.getaddrinfo(GEMINI, 443)
    assert fake_network.resolver.calls == ["html.duckduckgo.com"]
    assert logged(guarded) == [
        ("html.duckduckgo.com", 443, "allowed", "duckduckgo"),
        (GEMINI, 443, "blocked", "other"),
    ]


def test_with_a_proxy_only_the_proxy_host_is_allowed_in_private_mode(fake_network):
    made = make_guard(EgressPolicy(normal=frozenset(), private=frozenset({"tor.lan"})), private=True)
    made.guard.install()
    try:
        socket.getaddrinfo("tor.lan", 9050)
        with pytest.raises(EgressBlocked):
            socket.getaddrinfo("html.duckduckgo.com", 443)
    finally:
        made.guard.uninstall()
    assert fake_network.resolver.calls == ["tor.lan"]


def test_loopback_connections_work_in_private_mode(guarded):
    guarded.mode.private = True
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        with socket.create_connection(("localhost", port), timeout=5) as client:
            client.sendall(b"ping")
            accepted, _ = server.accept()
            with accepted:
                assert accepted.recv(4) == b"ping"
    entries = guarded.log.entries()
    assert {e.verdict for e in entries} == {"allowed"}
    assert entries[-1].host == "localhost"


def test_asyncio_connections_are_guarded(guarded):
    async def attempt():
        server = await asyncio.start_server(lambda reader, writer: writer.close(), "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            _, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            await asyncio.open_connection("203.0.113.5", 443)
        finally:
            server.close()

    guarded.mode.private = True
    with pytest.raises(EgressBlocked):
        asyncio.run(asyncio.wait_for(attempt(), 5))
    assert ("203.0.113.5", 443, "blocked", "other") in logged(guarded)
    assert any(host == "127.0.0.1" and verdict == "allowed" for host, _, verdict, _ in logged(guarded))


def test_http_clients_report_the_block_in_their_exception_chain(guarded):
    with httpx.Client(trust_env=False) as client, pytest.raises(httpx.ConnectError) as error:
        client.get("http://blocked.invalid/")
    block = egress_block_in(error.value)
    assert isinstance(block, EgressBlocked)
    assert block.host == "blocked.invalid"


def test_uninstall_restores_the_patched_functions():
    before = (
        socket.getaddrinfo,
        socket.socket.connect,
        socket.socket.connect_ex,
        proactor_events.BaseProactorEventLoop.sock_connect,
    )
    made = make_guard()
    made.guard.install()
    try:
        assert made.guard.installed
        assert socket.getaddrinfo is not before[0]
    finally:
        made.guard.uninstall()
    after = (
        socket.getaddrinfo,
        socket.socket.connect,
        socket.socket.connect_ex,
        proactor_events.BaseProactorEventLoop.sock_connect,
    )
    assert after == before
    assert "connect" not in vars(socket.socket)
    made.guard.uninstall()
    assert not made.guard.installed


def test_only_one_guard_can_be_installed(guarded):
    with pytest.raises(RuntimeError, match="already installed"):
        make_guard().guard.install()


def test_blocked_is_an_os_error_so_clients_treat_it_as_a_failed_connection():
    assert issubclass(EgressBlocked, OSError)


def test_egress_block_in_follows_causes_and_contexts():
    block = EgressBlocked("example.com", 443)
    wrapped = RuntimeError("wrapped")
    wrapped.__cause__ = ValueError("middle")
    wrapped.__cause__.__context__ = block
    assert egress_block_in(wrapped) is block
    assert egress_block_in(RuntimeError("unrelated")) is None
    assert egress_block_in(None) is None
