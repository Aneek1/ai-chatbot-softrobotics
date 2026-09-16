import asyncio
import logging
import threading
import uuid
from collections.abc import Callable, Iterator
from typing import Literal

import anyio
from fastapi import APIRouter, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.app.chat_service import Event
from backend.app.services import Services

logger = logging.getLogger(__name__)

router = APIRouter()

DISCONNECT_POLL_SECONDS = 0.5

# The event loop keeps only weak references to tasks.
_watchers: set[asyncio.Task] = set()


class ClosableEvents:
    """Lets one thread pull events while another closes the generator without racing it.

    `on_finish` runs exactly once: when the events run out, raise, or are closed.
    """

    def __init__(self, events: Iterator[Event], on_finish: Callable[[], None] = lambda: None):
        self._events = events
        self._on_finish = on_finish
        self._lock = threading.Lock()
        self.finished = False

    def _finish(self) -> None:
        if not self.finished:
            self.finished = True
            self._on_finish()

    def next(self) -> Event | None:
        with self._lock:
            if self.finished:
                return None
            try:
                item = next(self._events, None)
            except BaseException:
                self._finish()
                raise
            if item is None:
                self._finish()
            return item

    def close(self) -> None:
        with self._lock:
            try:
                self._events.close()
            finally:
                self._finish()


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
    chat_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    model: Literal["ollama", "gemini"] = "ollama"
    # A GlotLID label from the override menu; free text would reach the system prompt.
    language_override: str | None = Field(default=None, pattern=r"^[a-z]{3}_[A-Z][a-z]{3}$")


def answer_events(services: Services, body: ChatRequest, private: bool, chat_id: str) -> Iterator[Event]:
    # Private chats go to the in-memory store, so nothing from them reaches the history file.
    store = services.private_history if private else services.history
    yield from services.chat.run(
        body.message,
        body.model,
        body.language_override,
        store.detections(chat_id),
        private=private,
        record=lambda exchange: store.add_exchange(chat_id, exchange),
    )


@router.post("/api/chat", response_class=EventSourceResponse)
async def chat(body: ChatRequest, request: Request):
    services = request.app.state.services
    # Waits while a mode switch is pending; the answer then keeps this mode until it ends.
    private = await run_in_threadpool(services.mode.begin_answer)
    chat_id = body.chat_id or uuid.uuid4().hex
    events = ClosableEvents(
        answer_events(services, body, private, chat_id), on_finish=services.mode.end_answer
    )
    watcher = asyncio.create_task(close_on_disconnect(request, events))
    _watchers.add(watcher)
    watcher.add_done_callback(_watchers.discard)
    try:
        # Answer models stream with blocking I/O, so iterate them off the event loop.
        while (item := await run_in_threadpool(events.next)) is not None:
            event, data = item
            if event == "done":
                data = {**data, "chat_id": chat_id}
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
        if not events.finished:
            # Reached when this stream is cancelled mid-answer. Closing waits for the worker
            # thread to leave next(), so it runs on its own thread instead of the event loop.
            threading.Thread(target=events.close, daemon=True).start()
