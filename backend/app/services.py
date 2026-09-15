from dataclasses import dataclass

from backend.app.chat_service import ChatService
from backend.app.config import Settings
from backend.pipeline.embeddings import E5Embedder
from backend.pipeline.langid import Detection, FastTextScorer, TwoStageDetector
from backend.pipeline.retrieve import ChunkIndex
from backend.pipeline.specialist import load_specialist
from backend.providers.base import AnswerModel
from backend.providers.gemini import GeminiModel
from backend.providers.ollama import OllamaModel


@dataclass
class Services:
    settings: Settings
    chat: ChatService
    detections: dict[str, list[Detection]]
    ollama: OllamaModel | None
    index: ChunkIndex | None
    detector_name: str


def build_services(settings: Settings) -> Services:
    embedder = E5Embedder(settings.models_dir / settings.e5_dir)
    specialist = (
        load_specialist(settings.models_dir, settings.langid_specialist, embedder)
        if settings.langid_specialist
        else None
    )
    detector = TwoStageDetector(
        general=FastTextScorer(settings.models_dir / settings.glotlid_file),
        min_confidence=settings.langid_min_confidence,
        specialist=specialist,
    )
    index = ChunkIndex.open(settings.index_dir, embedder)
    ollama = OllamaModel(settings.ollama_url, settings.ollama_model, settings.ollama_timeout)
    models: dict[str, AnswerModel] = {"ollama": ollama}
    if settings.gemini_configured:
        models["gemini"] = GeminiModel(
            settings.google_api_key.get_secret_value(), settings.gemini_model, settings.gemini_timeout
        )
    chat = ChatService(detector=detector, retriever=index, models=models, top_k=settings.retrieval_top_k)
    return Services(
        settings=settings,
        chat=chat,
        detections={},
        ollama=ollama,
        index=index,
        detector_name=(
            settings.glotlid_file
            if specialist is None
            else f"{settings.glotlid_file} + {settings.langid_specialist}"
        ),
    )
