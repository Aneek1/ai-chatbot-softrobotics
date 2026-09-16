import logging
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from backend.pipeline.answer_language import choose_answer_language
from backend.pipeline.citations import check_citations
from backend.pipeline.langid import Detection, LanguageDetector
from backend.pipeline.languages import display_name, script_of_label
from backend.pipeline.prompt import build_messages
from backend.pipeline.retrieve import Hit, Retriever
from backend.providers.base import AnswerModel, ProviderError
from backend.search.base import SearchError, WebResult, WebSearch
from backend.store.history import Exchange

logger = logging.getLogger(__name__)

Event = tuple[str, Any]

SEARCH_OFF_NORMAL = (
    "Web search is off: set GOOGLE_SEARCH_KEY and GOOGLE_SEARCH_ENGINE_ID, or set WEB_SEARCH=duckduckgo"
)
SEARCH_OFF_PRIVATE = "Web search is off in private mode"


def detection_payload(detection: Detection) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "code": c.code,
                "name": display_name(c.code),
                "script": script_of_label(c.code),
                "probability": round(c.probability, 4),
            }
            for c in detection.candidates
        ],
        "chosen": detection.chosen,
        "chosen_name": display_name(detection.chosen),
        "uncertain": detection.uncertain,
        "stage": detection.stage,
    }


def sources_payload(hits: list[Hit]) -> list[dict[str, Any]]:
    return [
        {
            "id": h.chunk.id,
            "title": h.chunk.title,
            "snippet": h.chunk.text[:300],
            "language": h.chunk.language,
            "source": h.chunk.source,
            "url": h.chunk.url,
            "licence": h.chunk.licence,
        }
        for h in hits
    ]


def web_payload(results: list[WebResult]) -> list[dict[str, str]]:
    return [{"title": r.title, "url": r.url, "snippet": r.snippet, "engine": r.engine} for r in results]


@dataclass
class ChatService:
    detector: LanguageDetector
    retriever: Retriever
    models: Mapping[str, AnswerModel]
    top_k: int
    normal_search: WebSearch | None = None
    private_search: WebSearch | None = None
    max_web_results: int = 5

    def run(
        self,
        message: str,
        model_name: str,
        override: str | None,
        history: list[Detection],
        private: bool = False,
        record: Callable[[Exchange], None] | None = None,
    ) -> Iterator[Event]:
        """Yield (event, data) pairs in the order the API streams them. Appends the detection to history.

        A private answer uses only the private search and Ollama; it never falls back to the normal
        search or to Gemini. `record` receives a finished exchange; failed answers are not recorded.
        """
        detection = self.detector.detect(message)
        language = detection_payload(detection)
        yield "language", language
        answer_language = choose_answer_language(detection, override, history)
        history.append(detection)

        hits = self.retriever.search(message, self.top_k)
        sources = sources_payload(hits)
        yield "sources", sources

        web, notice = self._search(message, private)
        web_items = web_payload(web)
        yield "web", web_items
        if notice is not None:
            yield "notice", notice
        if not hits and not web:
            yield "notice", {"kind": "no_sources", "message": "No sources found"}

        if private and model_name != "ollama":
            yield "notice", {"kind": "private_mode_model", "message": "Private mode answers with Ollama only"}
            model_name = "ollama"
        model = self.models.get(model_name)
        if model is None:
            if private:
                detail = "Private mode needs Ollama, which is not configured"
                yield "error", {"code": "ollama_unavailable", "message": detail}
            else:
                detail = f"Answer model '{model_name}' is not configured"
                yield "error", {"code": "model_unavailable", "message": detail}
            return

        pieces: list[str] = []
        stream = model.stream(build_messages(message, answer_language, hits, web))
        try:
            for piece in stream:
                pieces.append(piece)
                yield "token", {"text": piece}
        except ProviderError as exc:
            yield "error", {"code": exc.code, "message": str(exc)}
            return
        finally:
            # Closing this generator early (the client left) must also end the provider's request.
            # AnswerModel only promises an iterator; generators are the ones holding a request open.
            close = getattr(stream, "close", None)
            if close is not None:
                close()

        # Web results are never citable, so only retrieved sources count as valid numbers.
        check = check_citations("".join(pieces), source_count=len(hits))
        citations = {"valid": list(check.valid), "removed": list(check.removed)}
        yield "citations", citations
        if record is not None:
            record(
                Exchange(
                    question=message,
                    detection=language,
                    answer=check.text,
                    answer_language=answer_language,
                    sources=sources,
                    web=web_items,
                    citations=citations,
                )
            )
        yield "done", {"answer_language": answer_language}

    def _search(self, message: str, private: bool) -> tuple[list[WebResult], dict[str, str] | None]:
        search = self.private_search if private else self.normal_search
        if search is None:
            return [], {
                "kind": "web_search_off",
                "message": SEARCH_OFF_PRIVATE if private else SEARCH_OFF_NORMAL,
            }
        try:
            return search.search(message, self.max_web_results), None
        except SearchError as exc:
            return [], {"kind": exc.code, "message": str(exc)}
        except Exception as exc:
            # Only the type is logged: exception text can contain the question, which stays out of logs.
            logger.warning("web search failed with %s", type(exc).__name__)
            return [], {"kind": "web_search_failed", "message": "Web search failed"}
