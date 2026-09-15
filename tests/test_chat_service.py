from backend.app.chat_service import ChatService
from backend.pipeline.langid import TwoStageDetector
from backend.pipeline.retrieve import Chunk, Hit
from backend.providers.base import ProviderError
from tests.fakes import FakeModel, FakeScorer


class FakeRetriever:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, top_k):
        return self._hits[:top_k]


def hit(n):
    chunk = Chunk(
        f"d{n}:0", f"d{n}", f"text {n}", "eng_Latn", "fixture",
        f"T{n}", "https://example.invalid", "test-only",
    )
    return Hit(chunk, 0.8)


def service(hits, model=None, candidates=(("ind_Latn", 0.97),)):
    return ChatService(
        detector=TwoStageDetector(general=FakeScorer(list(candidates)), min_confidence=0.6),
        retriever=FakeRetriever(hits),
        models={"fake": model or FakeModel(pieces=("Jawaban", " [1]", " [5]"))},
        top_k=6,
    )


def run(svc, history=None, model="fake", override=None):
    history = history if history is not None else []
    return list(svc.run("Bagaimana cara membuat aktuator?", model, override, history))


def test_event_order_and_payloads():
    events = run(service([hit(1), hit(2)]))
    assert [e for e, _ in events] == ["language", "sources", "token", "token", "token", "citations", "done"]

    language = events[0][1]
    assert language["chosen"] == "ind_Latn"
    assert language["chosen_name"] == "Indonesian"
    assert language["candidates"][0] == {
        "code": "ind_Latn", "name": "Indonesian", "script": "Latn", "probability": 0.97
    }

    assert events[1][1][0]["title"] == "T1"
    assert events[-2][1] == {"valid": [1], "removed": [5]}
    assert events[-1][1] == {"answer_language": "ind_Latn"}


def test_answer_prompt_uses_detected_language():
    model = FakeModel()
    run(service([hit(1)], model=model))
    assert "Answer in Indonesian." in model.calls[0][0].content


def test_no_sources_adds_a_notice():
    events = run(service([]))
    assert ("notice", {"kind": "no_sources", "message": "No sources found"}) in events


def test_provider_error_ends_with_error_event():
    model = FakeModel(error=ProviderError("ollama_unavailable", "Ollama is not reachable"))
    events = run(service([hit(1)], model=model))
    assert events[-1] == ("error", {"code": "ollama_unavailable", "message": "Ollama is not reachable"})
    assert "done" not in [e for e, _ in events]


def test_unknown_model_is_an_error():
    events = run(service([hit(1)]), model="gemini")
    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "model_unavailable"


def test_detection_is_appended_to_history():
    history = []
    run(service([hit(1)]), history=history)
    assert [d.chosen for d in history] == ["ind_Latn"]
