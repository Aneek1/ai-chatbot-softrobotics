import socket
import threading
from asyncio import proactor_events
from collections.abc import Callable, Mapping

from backend.privacy.egress_log import EgressLog, Verdict
from backend.privacy.policy import LOCAL_NAMES, EgressPolicy, component_for, normalize_host

_INET = (socket.AF_INET, socket.AF_INET6)
_MISSING = object()
_install_lock = threading.Lock()
_active: "EgressGuard | None" = None


class EgressBlocked(PermissionError):
    """Raised in place of a DNS lookup or connection that the current mode does not allow.

    It is an OSError, so HTTP clients report it as a failed connection and keep it in the chain.
    """

    def __init__(self, host: str, port: int | None):
        super().__init__(f"The egress guard blocked a connection to {host}:{port}")
        self.host = host
        self.port = port


def egress_block_in(exc: BaseException | None) -> EgressBlocked | None:
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, EgressBlocked):
            return exc
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return None


def _port(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str | bytes) and value.isdigit():
        return int(value)
    return None


class EgressGuard:
    """Checks every DNS lookup and socket connection in this process against the policy."""

    def __init__(
        self,
        policy: EgressPolicy,
        is_private: Callable[[], bool],
        log: EgressLog,
        labels: Mapping[str, str] | None = None,
    ):
        self.policy = policy
        self.log = log
        self._is_private = is_private
        self._labels = dict(labels or {})
        self._resolved: dict[str, str] = {}
        self._lock = threading.Lock()
        self._originals: list[tuple[object, str, object]] | None = None

    @property
    def installed(self) -> bool:
        return self._originals is not None

    def install(self) -> None:
        global _active
        with _install_lock:
            if _active is not None:
                raise RuntimeError("An egress guard is already installed in this process")
            real_getaddrinfo = socket.getaddrinfo
            real_connect = socket.socket.connect
            real_connect_ex = socket.socket.connect_ex
            real_sock_connect = proactor_events.BaseProactorEventLoop.sock_connect
            guard = self

            def getaddrinfo(host, port, *args, **kwargs):
                guard._check_lookup(host, port)
                results = real_getaddrinfo(host, port, *args, **kwargs)
                guard._remember(host, results)
                return results

            def connect(sock, address):
                guard._check_connect(sock, address)
                return real_connect(sock, address)

            def connect_ex(sock, address):
                guard._check_connect(sock, address)
                return real_connect_ex(sock, address)

            # asyncio's Windows event loop connects with ConnectEx and never calls socket.connect.
            async def sock_connect(loop, sock, address):
                guard._check_connect(sock, address)
                return await real_sock_connect(loop, sock, address)

            self._originals = []
            for owner, name, replacement in (
                (socket, "getaddrinfo", getaddrinfo),
                (socket.socket, "connect", connect),
                (socket.socket, "connect_ex", connect_ex),
                (proactor_events.BaseProactorEventLoop, "sock_connect", sock_connect),
            ):
                # socket.socket inherits connect from _socket.socket; remembering that its own
                # attribute was missing lets uninstall delete ours instead of leaving a copy.
                self._originals.append((owner, name, vars(owner).get(name, _MISSING)))
                setattr(owner, name, replacement)
            _active = self

    def uninstall(self) -> None:
        global _active
        with _install_lock:
            if _active is not self or self._originals is None:
                return
            for owner, name, original in reversed(self._originals):
                if original is _MISSING:
                    delattr(owner, name)
                else:
                    setattr(owner, name, original)
            self._originals = None
            _active = None

    def _record(self, host: str, port: int | None, verdict: Verdict) -> None:
        self.log.record(host, port, verdict, component_for(self._labels, host, port))

    def _check_lookup(self, host: str | bytes | None, port: object) -> None:
        if host is None:
            return
        name = normalize_host(host)
        if not self.policy.allows(name, self._is_private()):
            self._record(name, _port(port), "blocked")
            raise EgressBlocked(name, _port(port))

    def _remember(self, host: str | bytes | None, results: list) -> None:
        if host is None:
            return
        name = normalize_host(host)
        with self._lock:
            for family, _type, _proto, _canonname, sockaddr in results:
                if family in _INET:
                    self._resolved[normalize_host(sockaddr[0])] = name

    def _check_connect(self, sock: socket.socket, address: object) -> None:
        if sock.family not in _INET or not isinstance(address, tuple):
            return
        target = normalize_host(address[0])
        port = _port(address[1])
        with self._lock:
            name = self._resolved.get(target, target)
        private = self._is_private()
        # "localhost" only counts when the address really is loopback.
        allowed = self.policy.allows(target, private) or (
            name not in LOCAL_NAMES and self.policy.allows(name, private)
        )
        self._record(name, port, "allowed" if allowed else "blocked")
        if not allowed:
            raise EgressBlocked(name, port)
