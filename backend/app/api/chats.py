from fastapi import APIRouter, HTTPException, Request, Response

from backend.store.history import SqliteHistory

router = APIRouter()


def current_store(request: Request) -> SqliteHistory:
    services = request.app.state.services
    # While private mode is on, only its in-memory chats are visible.
    return services.private_history if services.mode.private else services.history


@router.get("/api/chats")
def list_chats(request: Request) -> list[dict]:
    return current_store(request).list_chats()


@router.get("/api/chats/{chat_id}")
def get_chat(chat_id: str, request: Request) -> dict:
    chat = current_store(request).get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.delete("/api/chats/{chat_id}", status_code=204)
def delete_chat(chat_id: str, request: Request) -> Response:
    if not current_store(request).delete_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    return Response(status_code=204)
