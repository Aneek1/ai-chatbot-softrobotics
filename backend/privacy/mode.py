import threading
from collections.abc import Callable


class ModeSwitchTimeout(Exception):
    pass


class ModeState:
    """The app-wide private switch. A switch waits until no answer is being generated."""

    def __init__(self, private: bool = False):
        self._private = private
        self._in_flight = 0
        self._switching = False
        self._cond = threading.Condition()
        self._listeners: list[Callable[[bool], None]] = []

    @property
    def private(self) -> bool:
        return self._private

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def switching(self) -> bool:
        return self._switching

    def on_change(self, listener: Callable[[bool], None]) -> None:
        """Listeners run while the switch still holds new answers back."""
        self._listeners.append(listener)

    def begin_answer(self) -> bool:
        """Wait out a pending switch, then return the mode this answer uses from start to finish."""
        with self._cond:
            self._cond.wait_for(lambda: not self._switching)
            self._in_flight += 1
            return self._private

    def end_answer(self) -> None:
        with self._cond:
            if self._in_flight == 0:
                raise RuntimeError("end_answer called without begin_answer")
            self._in_flight -= 1
            self._cond.notify_all()

    def set_private(self, private: bool, timeout: float) -> bool:
        """Return True if the mode changed. Raise ModeSwitchTimeout if answers are still running."""
        with self._cond:
            if not self._cond.wait_for(lambda: not self._switching, timeout):
                raise ModeSwitchTimeout("Another mode switch is still waiting; mode unchanged")
            if private == self._private:
                return False
            self._switching = True
            try:
                if not self._cond.wait_for(lambda: self._in_flight == 0, timeout):
                    raise ModeSwitchTimeout(
                        f"{self._in_flight} answer(s) still running after {timeout} s; mode unchanged"
                    )
                self._private = private
                for listener in self._listeners:
                    listener(private)
            finally:
                self._switching = False
                self._cond.notify_all()
            return True
