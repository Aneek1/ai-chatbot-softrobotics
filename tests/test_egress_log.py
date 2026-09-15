import threading

from backend.privacy.egress_log import EgressLog


def fixed_clock():
    return "2026-09-16T00:00:00.000+00:00"


def test_records_entries_in_order_with_sequence_numbers():
    log = EgressLog(clock=fixed_clock)
    first = log.record("localhost", 11434, "allowed", "ollama")
    second = log.record("example.com", 443, "blocked", "other")
    assert (first.seq, second.seq) == (1, 2)
    assert [e.payload() for e in log.entries()] == [
        {
            "time": "2026-09-16T00:00:00.000+00:00",
            "host": "localhost",
            "port": 11434,
            "verdict": "allowed",
            "component": "ollama",
        },
        {
            "time": "2026-09-16T00:00:00.000+00:00",
            "host": "example.com",
            "port": 443,
            "verdict": "blocked",
            "component": "other",
        },
    ]


def test_keeps_only_the_newest_entries():
    log = EgressLog(capacity=3)
    for port in range(5):
        log.record("localhost", port, "allowed", "other")
    assert [e.port for e in log.entries()] == [2, 3, 4]
    assert [e.seq for e in log.entries()] == [3, 4, 5]


def test_default_capacity_is_500():
    log = EgressLog()
    for port in range(501):
        log.record("localhost", port, "allowed", "other")
    assert len(log.entries()) == 500
    assert log.entries()[0].port == 1


def test_since_returns_only_newer_entries():
    log = EgressLog()
    for port in (1, 2, 3):
        log.record("localhost", port, "allowed", "other")
    assert [e.port for e in log.since(1)] == [2, 3]
    assert log.since(3) == []


def test_default_time_is_utc():
    entry = EgressLog().record("localhost", 1, "allowed", "other")
    assert entry.time.endswith("+00:00")


def test_concurrent_records_get_unique_sequence_numbers():
    log = EgressLog(capacity=1000)

    def work():
        for port in range(100):
            log.record("localhost", port, "allowed", "other")

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(e.seq for e in log.entries()) == list(range(1, 801))
