from fastapi import APIRouter, Request

from backend.app.config import Settings
from backend.search.base import WebSearch

router = APIRouter()


def search_name(search: WebSearch | None) -> str | None:
    return search.name if search is not None else None


def model_files(settings: Settings) -> dict[str, bool]:
    present = {
        "glotlid": (settings.models_dir / settings.glotlid_file).is_file(),
        "e5": (settings.models_dir / settings.e5_dir).is_dir(),
    }
    if settings.langid_specialist:
        present["specialist"] = (settings.models_dir / settings.langid_specialist).is_file()
    return present


@router.get("/api/health")
def health(request: Request, warm: bool = False) -> dict:
    services = request.app.state.services
    ollama_reachable = False
    if services.ollama is not None:
        ollama_reachable = services.ollama.warm() if warm else services.ollama.reachable()
    return {
        "detector": services.detector_name,
        "model_files": model_files(services.settings),
        "ollama_reachable": ollama_reachable,
        "gemini_configured": services.settings.gemini_configured,
        "web_search": {
            "normal": search_name(services.chat.normal_search),
            "private": search_name(services.chat.private_search),
        },
        "private": services.mode.private,
        "egress_guard": services.guard is not None and services.guard.installed,
        "documents_by_language": services.index.count_by_language() if services.index else {},
    }
