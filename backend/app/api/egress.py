import anyio
from fastapi import APIRouter, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent

router = APIRouter()

POLL_SECONDS = 0.5


def last_seen(request: Request) -> int:
    value = request.headers.get("last-event-id", "")
    return int(value) if value.isdigit() else 0


@router.get("/api/egress", response_class=EventSourceResponse)
async def egress(request: Request, follow: bool = True):
    """Stream logged connection attempts. `follow=false` sends what is logged now and ends."""
    log = request.app.state.services.egress_log
    last = last_seen(request)
    while True:
        for entry in log.since(last):
            last = entry.seq
            yield ServerSentEvent(event="connection", id=str(entry.seq), data=entry.payload())
        if not follow or await request.is_disconnected():
            return
        await anyio.sleep(POLL_SECONDS)
