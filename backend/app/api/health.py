from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/api/health")
def health(request: Request, warm: bool = False) -> dict:
    services = request.app.state.services
    ollama_reachable = False
    if services.ollama is not None:
        ollama_reachable = services.ollama.warm() if warm else services.ollama.reachable()
    return {
        "detector": services.detector_name,
        "ollama_reachable": ollama_reachable,
        "gemini_configured": services.settings.gemini_configured,
        "documents_by_language": services.index.count_by_language() if services.index else {},
    }
