from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


class ProviderError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class AnswerModel(Protocol):
    name: str

    def stream(self, messages: Sequence[Message]) -> Iterator[str]: ...
