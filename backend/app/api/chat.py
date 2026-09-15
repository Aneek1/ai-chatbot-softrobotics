import logging
from typing import Literal

from fastapi import APIRouter, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool

logger = logging.getLogger(__name__)

router = APIRouter()


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
    events = services.chat.run(body.message, body.model, body.language_override, history)
    # Answer models stream with blocking I/O, so iterate them off the event loop.
    try:
        async for event, data in iterate_in_threadpool(events):
            yield ServerSentEvent(event=event, data=data)
    except Exception:
        # The 200 status is already sent, so every stream must still end with `done` or `error`.
        # The exception text stays in the log because it can reveal internal details.
        logger.exception("chat stream failed")
        yield ServerSentEvent(
            event="error",
            data={"code": "internal_error", "message": "The server hit an unexpected error"},
        )
