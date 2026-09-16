from dataclasses import dataclass, field

from backend.app.chat_service import ChatService
from backend.app.config import Settings
from backend.pipeline.embeddings import E5Embedder
from backend.pipeline.langid import FastTextScorer, TwoStageDetector
from backend.pipeline.retrieve import ChunkIndex
from backend.pipeline.specialist import load_specialist
from backend.privacy.egress import EgressGuard
from backend.privacy.egress_log import EgressLog
from backend.privacy.mode import ModeState
from backend.privacy.policy import build_labels, build_policy
from backend.providers.base import AnswerModel
from backend.providers.gemini import GeminiModel
from backend.providers.ollama import OllamaModel
from backend.search.base import WebSearch
from backend.search.duckduckgo import DuckDuckGoSearch
from backend.search.google import GoogleSearch
from backend.store.history import SqliteHistory


@dataclass
class Privacy:
    mode: ModeState
    egress_log: EgressLog
    guard: EgressGuard


@dataclass
class Services:
    settings: Settings
    chat: ChatService
    ollama: OllamaModel | None
    index: ChunkIndex | None
    detector_name: str
    mode: ModeState = field(default_factory=ModeState)
    egress_log: EgressLog = field(default_factory=EgressLog)
    guard: EgressGuard | None = None
    history: SqliteHistory = field(default_factory=SqliteHistory.in_memory)
    private_history: SqliteHistory = field(default_factory=SqliteHistory.in_memory)

    def __post_init__(self) -> None:
        self.mode.on_change(self._mode_changed)

    def _mode_changed(self, private: bool) -> None:
        # Private chats exist only until private mode ends.
        if not private:
            self.private_history.clear()

    def close(self) -> None:
        self.history.close()
        self.private_history.close()
        if self.index is not None:
            self.index.close()


def build_privacy(settings: Settings) -> Privacy:
    mode = ModeState(private=settings.private_mode)
    log = EgressLog()
    guard = EgressGuard(build_policy(settings), lambda: mode.private, log, build_labels(settings))
    return Privacy(mode=mode, egress_log=log, guard=guard)


def build_normal_search(settings: Settings) -> WebSearch | None:
    if settings.web_search == "duckduckgo":
        return DuckDuckGoSearch(timeout=settings.search_timeout)
    if settings.web_search == "google" and settings.google_search_configured:
        return GoogleSearch(
            settings.google_search_key.get_secret_value(),
            settings.google_search_engine_id,
            timeout=settings.search_timeout,
        )
    return None


def build_services(settings: Settings, privacy: Privacy | None = None) -> Services:
    privacy = privacy or build_privacy(settings)
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
    chat = ChatService(
        detector=detector,
        retriever=index,
        models=models,
        top_k=settings.retrieval_top_k,
        normal_search=build_normal_search(settings),
        private_search=DuckDuckGoSearch(timeout=settings.search_timeout, proxy=settings.private_proxy),
        max_web_results=settings.search_max_results,
    )
    return Services(
        settings=settings,
        chat=chat,
        ollama=ollama,
        index=index,
        detector_name=(
            settings.glotlid_file
            if specialist is None
            else f"{settings.glotlid_file} + {settings.langid_specialist}"
        ),
        mode=privacy.mode,
        egress_log=privacy.egress_log,
        guard=privacy.guard,
        history=SqliteHistory.open(settings.history_db),
        private_history=SqliteHistory.in_memory(),
    )
