import threading
import time

import pytest

from backend.privacy.mode import ModeState, ModeSwitchFailed, ModeSwitchTimeout


def wait_until(condition, seconds=5.0):
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.01)


def test_starts_in_the_configured_mode():
    assert ModeState().private is False
    assert ModeState(private=True).private is True


def test_switch_applies_at_once_when_no_answer_is_running():
    mode = ModeState()
    assert mode.set_private(True, timeout=1) is True
    assert mode.private is True
    assert mode.set_private(True, timeout=1) is False


def test_switch_waits_for_in_flight_answers():
    mode = ModeState()
    assert mode.begin_answer() is False
    thread = threading.Thread(target=mode.set_private, args=(True, 5))
    thread.start()
    wait_until(lambda: mode.switching)
    time.sleep(0.1)
    assert mode.private is False

    mode.end_answer()
    thread.join(5)
    assert mode.private is True
    assert mode.switching is False


def test_answers_that_start_during_a_switch_wait_and_use_the_new_mode():
    mode = ModeState()
    mode.begin_answer()
    switch = threading.Thread(target=mode.set_private, args=(True, 5))
    switch.start()
    wait_until(lambda: mode.switching)

    seen = []

    def answer():
        seen.append(mode.begin_answer())
        mode.end_answer()

    waiting = threading.Thread(target=answer)
    waiting.start()
    time.sleep(0.1)
    assert seen == []

    mode.end_answer()
    waiting.join(5)
    switch.join(5)
    assert seen == [True]


def test_switch_times_out_and_keeps_the_old_mode():
    mode = ModeState()
    mode.begin_answer()
    with pytest.raises(ModeSwitchTimeout, match="mode unchanged"):
        mode.set_private(True, timeout=0.05)
    assert mode.private is False
    assert mode.switching is False
    assert mode.begin_answer() is False
    mode.end_answer()
    mode.end_answer()
    assert mode.in_flight == 0


def test_listeners_run_only_when_the_mode_changes():
    mode = ModeState()
    calls = []
    mode.on_change(calls.append)
    mode.set_private(True, timeout=1)
    mode.set_private(True, timeout=1)
    mode.set_private(False, timeout=1)
    assert calls == [True, False]


def test_a_listener_that_raises_keeps_the_old_mode():
    mode = ModeState()

    def boom(private):
        raise ValueError("boom")

    mode.on_change(boom)
    with pytest.raises(ModeSwitchFailed, match="mode unchanged") as caught:
        mode.set_private(True, timeout=1)
    assert [str(error) for error in caught.value.errors] == ["boom"]
    assert mode.private is False
    assert mode.switching is False
    assert mode.begin_answer() is False
    mode.end_answer()


def test_every_listener_runs_even_when_one_raises():
    mode = ModeState()
    calls = []

    def boom(private):
        calls.append("boom")
        raise ValueError("boom")

    mode.on_change(lambda private: calls.append("first"))
    mode.on_change(boom)
    mode.on_change(lambda private: calls.append("last"))
    with pytest.raises(ModeSwitchFailed):
        mode.set_private(True, timeout=1)
    assert calls == ["first", "boom", "last"]


def test_end_answer_without_begin_is_an_error():
    with pytest.raises(RuntimeError):
        ModeState().end_answer()
