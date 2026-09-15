from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from backend.pipeline.answer_language import choose_answer_language
from backend.pipeline.citations import check_citations
from backend.pipeline.langid import Detection, LanguageDetector
from backend.pipeline.languages import display_name, script_of_label
from backend.pipeline.prompt import build_messages
from backend.pipeline.retrieve import Hit, Retriever
from backend.providers.base import AnswerModel, ProviderError

Event = tuple[str, Any]


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


@dataclass
class ChatService:
    detector: LanguageDetector
    retriever: Retriever
    models: Mapping[str, AnswerModel]
    top_k: int

    def run(
        self, message: str, model_name: str, override: str | None, history: list[Detection]
    ) -> Iterator[Event]:
        """Yield (event, data) pairs in the order the API streams them. Appends the detection to history."""
        detection = self.detector.detect(message)
        yield "language", detection_payload(detection)
        answer_language = choose_answer_language(detection, override, history)
        history.append(detection)

        hits = self.retriever.search(message, self.top_k)
        yield "sources", sources_payload(hits)
        if not hits:
            yield "notice", {"kind": "no_sources", "message": "No sources found"}

        model = self.models.get(model_name)
        if model is None:
            detail = f"Answer model '{model_name}' is not configured"
            yield "error", {"code": "model_unavailable", "message": detail}
            return

        pieces: list[str] = []
        try:
            for piece in model.stream(build_messages(message, answer_language, hits)):
                pieces.append(piece)
                yield "token", {"text": piece}
        except ProviderError as exc:
            yield "error", {"code": exc.code, "message": str(exc)}
            return

        check = check_citations("".join(pieces), source_count=len(hits))
        yield "citations", {"valid": list(check.valid), "removed": list(check.removed)}
        yield "done", {"answer_language": answer_language}
