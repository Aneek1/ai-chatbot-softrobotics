from typing import Literal

from fastapi import APIRouter, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    chat_id: str | None = None
    model: Literal["ollama", "gemini"] = "ollama"
    language_override: str | None = None


@router.post("/api/chat", response_class=EventSourceResponse)
async def chat(body: ChatRequest, request: Request):
    services = request.app.state.services
    history = services.detections.setdefault(body.chat_id, []) if body.chat_id else []
    events = services.chat.run(body.message, body.model, body.language_override, history)
    # Answer models stream with blocking I/O, so iterate them off the event loop.
    async for event, data in iterate_in_threadpool(events):
        yield ServerSentEvent(event=event, data=data)
