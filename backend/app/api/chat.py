import asyncio
import logging
import threading
from collections.abc import Iterator
from typing import Literal

import anyio
from fastapi import APIRouter, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.app.chat_service import Event

logger = logging.getLogger(__name__)

router = APIRouter()

DISCONNECT_POLL_SECONDS = 0.5

# The event loop keeps only weak references to tasks.
_watchers: set[asyncio.Task] = set()


class ClosableEvents:
    """Lets one thread pull events while another closes the generator without racing it."""

    def __init__(self, events: Iterator[Event]):
        self._events = events
        self._lock = threading.Lock()
        self.finished = False

    def next(self) -> Event | None:
        with self._lock:
            if self.finished:
                return None
            item = next(self._events, None)
            if item is None:
                self.finished = True
            return item

    def close(self) -> None:
        with self._lock:
            self.finished = True
            self._events.close()


async def close_on_disconnect(request: Request, events: ClosableEvents) -> None:
    # FastAPI stops pulling from a stream whose client left but never closes it, so the
    # answer model (and its HTTP request to Ollama) would stay open until garbage collection.
    while not events.finished:
        if await request.is_disconnected():
            await run_in_threadpool(events.close)
            return
        await anyio.sleep(DISCONNECT_POLL_SECONDS)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    chat_id: str | None = None
    model: Literal["ollama", "gemini"] = "ollama"
    # A GlotLID label from the override menu; free text would reach the system prompt.
    language_override: str | None = Field(default=None, pattern=r"^[a-z]{3}_[A-Z][a-z]{3}$")


@router.post("/api/chat", response_class=EventSourceResponse)
async def chat(body: ChatRequest, request: Request):
    services = request.app.state.services
    history = services.detections.setdefault(body.chat_id, []) if body.chat_id else []
    events = ClosableEvents(
        services.chat.run(body.message, body.model, body.language_override, history)
    )
    watcher = asyncio.create_task(close_on_disconnect(request, events))
    _watchers.add(watcher)
    watcher.add_done_callback(_watchers.discard)
    try:
        # Answer models stream with blocking I/O, so iterate them off the event loop.
        while (item := await run_in_threadpool(events.next)) is not None:
            event, data = item
            yield ServerSentEvent(event=event, data=data)
    except Exception:
        # The 200 status is already sent, so every stream must still end with `done` or `error`.
        # The exception text stays in the log because it can reveal internal details.
        logger.exception("chat stream failed")
        yield ServerSentEvent(
            event="error",
            data={"code": "internal_error", "message": "The server hit an unexpected error"},
        )
    finally:
        watcher.cancel()
