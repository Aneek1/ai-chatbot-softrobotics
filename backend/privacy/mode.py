import threading
from collections.abc import Callable


class ModeSwitchError(Exception):
    """A switch that did not happen. The mode is unchanged."""


class ModeSwitchTimeout(ModeSwitchError):
    pass


class ModeSwitchFailed(ModeSwitchError):
    """One or more listeners refused the new mode."""

    def __init__(self, message: str, errors: list[BaseException]):
        super().__init__(message)
        self.errors = errors


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
        """Listeners run with the new mode while the switch still holds new answers back.

        They run before the flag flips, so a listener that raises cancels the switch.
        """
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
        """Return True if the mode changed.

        Raise ModeSwitchTimeout if answers are still running, and ModeSwitchFailed if a
        listener raises. Both are ModeSwitchError and both leave the mode as it was: the
        flag flips only once every listener has accepted the new mode, and one listener
        raising never stops the rest from running. What a listener did before it raised is
        not undone, so listeners should leave the side of the switch they refuse usable.
        """
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
                errors: list[BaseException] = []
                for listener in self._listeners:
                    try:
                        listener(private)
                    except Exception as error:  # every listener gets a turn; all failures reported
                        errors.append(error)
                if errors:
                    raise ModeSwitchFailed(
                        f"{len(errors)} mode listener(s) refused the switch; mode unchanged",
                        errors,
                    )
                self._private = private
            finally:
                self._switching = False
                self._cond.notify_all()
            return True
