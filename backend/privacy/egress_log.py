import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

Verdict = Literal["allowed", "blocked"]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class EgressEntry:
    seq: int
    time: str
    host: str
    port: int | None
    verdict: Verdict
    component: str

    def payload(self) -> dict:
        return {
            "time": self.time,
            "host": self.host,
            "port": self.port,
            "verdict": self.verdict,
            "component": self.component,
        }


class EgressLog:
    """The most recent connection attempts, newest last, read by GET /api/egress."""

    def __init__(self, capacity: int = 500, clock: Callable[[], str] = _now):
        self._entries: deque[EgressEntry] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._seq = 0
        self._clock = clock

    def record(self, host: str, port: int | None, verdict: Verdict, component: str) -> EgressEntry:
        with self._lock:
            self._seq += 1
            entry = EgressEntry(self._seq, self._clock(), host, port, verdict, component)
            self._entries.append(entry)
        return entry

    def entries(self) -> list[EgressEntry]:
        with self._lock:
            return list(self._entries)

    def since(self, seq: int) -> list[EgressEntry]:
        with self._lock:
            return [e for e in self._entries if e.seq > seq]
