from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, StrictBool

from backend.app.services import Services
from backend.privacy.mode import ModeSwitchError

router = APIRouter()


class ModeRequest(BaseModel):
    private: StrictBool


def mode_state(services: Services) -> dict:
    return {
        "private": services.mode.private,
        "proxy_configured": services.settings.private_proxy is not None,
        "ollama_reachable": services.ollama.reachable() if services.ollama is not None else False,
    }


@router.get("/api/mode")
def get_mode(request: Request) -> dict:
    return mode_state(request.app.state.services)


@router.put("/api/mode")
def put_mode(body: ModeRequest, request: Request) -> dict:
    # A plain `def` route runs in the threadpool, so waiting for running answers does not block
    # the event loop. The switch happens even if Ollama is unreachable; chats then fail instead.
    services = request.app.state.services
    try:
        services.mode.set_private(body.private, timeout=services.settings.mode_switch_timeout)
    except ModeSwitchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return mode_state(services)
