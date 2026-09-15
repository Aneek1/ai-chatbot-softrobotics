import socket
from asyncio import proactor_events

import pytest


def _socket_hooks():
    return (
        socket.getaddrinfo,
        socket.socket.connect,
        socket.socket.connect_ex,
        proactor_events.BaseProactorEventLoop.sock_connect,
    )


@pytest.fixture(autouse=True)
def no_egress_guard_left_installed():
    """Fail the test that leaves socket functions patched, not a later test that trips over them."""
    before = _socket_hooks()
    yield
    assert _socket_hooks() == before, "a test left the egress guard (or another socket patch) installed"
    assert "connect" not in vars(socket.socket)
    assert "connect_ex" not in vars(socket.socket)
