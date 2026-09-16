from backend.app.chat_service import ChatService
from backend.pipeline.langid import TwoStageDetector
from backend.pipeline.retrieve import Chunk, Hit
from backend.providers.base import ProviderError
from backend.search.base import SearchError, WebResult
from tests.fakes import FakeModel, FakeScorer, FakeSearch

QUESTION = "Bagaimana cara membuat aktuator?"
WEB = WebResult("Gel grippers", "https://example.net/gel", "Gels swell.", "duckduckgo")
WEB_PAYLOAD = {
    "title": "Gel grippers",
    "url": "https://example.net/gel",
    "snippet": "Gels swell.",
    "engine": "duckduckgo",
}


class FakeRetriever:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, top_k):
        return self._hits[:top_k]


def hit(n):
    chunk = Chunk(
        f"d{n}:0",
        f"d{n}",
        f"text {n}",
        "eng_Latn",
        "fixture",
        f"T{n}",
        "https://example.invalid",
        "test-only",
    )
    return Hit(chunk, 0.8)


def service(hits, model=None, models=None, normal_search=None, private_search=None):
    if models is None:
        models = {"fake": model or FakeModel(pieces=("Jawaban", " [1]", " [5]"))}
    return ChatService(
        detector=TwoStageDetector(general=FakeScorer([("ind_Latn", 0.97)]), min_confidence=0.6),
        retriever=FakeRetriever(hits),
        models=models,
        top_k=6,
        normal_search=normal_search,
        private_search=private_search,
        max_web_results=5,
    )


def run(svc, history=None, model="fake", override=None, private=False, record=None):
    history = history if history is not None else []
    return list(svc.run(QUESTION, model, override, history, private=private, record=record))


def names(events):
    return [e for e, _ in events]


def notices(events):
    return [data for e, data in events if e == "notice"]


def test_event_order_and_payloads():
    events = run(service([hit(1), hit(2)], normal_search=FakeSearch([WEB])))
    assert names(events) == ["language", "sources", "web", "token", "token", "token", "citations", "done"]

    language = events[0][1]
    assert language["chosen"] == "ind_Latn"
    assert language["chosen_name"] == "Indonesian"
    assert language["candidates"][0] == {
        "code": "ind_Latn",
        "name": "Indonesian",
        "script": "Latn",
        "probability": 0.97,
    }

    assert events[1][1][0]["title"] == "T1"
    assert events[2][1] == [WEB_PAYLOAD]
    assert events[-2][1] == {"valid": [1], "removed": [5]}
    assert events[-1][1] == {"answer_language": "ind_Latn"}


def test_answer_prompt_uses_detected_language():
    model = FakeModel()
    run(service([hit(1)], model=model))
    assert "Answer in Indonesian." in model.calls[0][0].content


def test_web_results_reach_the_prompt_but_only_retrieved_sources_are_citable():
    model = FakeModel(pieces=("A [1]", " B [2]", " C [3]"))
    events = run(service([hit(1)], model=model, normal_search=FakeSearch([WEB, WEB])))
    assert "<web_results>" in model.calls[0][0].content
    assert ("citations", {"valid": [1], "removed": [2, 3]}) in events


def test_no_sources_notice_only_when_retrieval_and_web_both_find_nothing():
    empty = run(service([], normal_search=FakeSearch([])))
    assert {"kind": "no_sources", "message": "No sources found"} in notices(empty)

    web_only = run(service([], normal_search=FakeSearch([WEB])))
    assert "no_sources" not in [n["kind"] for n in notices(web_only)]


def test_missing_search_gives_a_web_search_off_notice():
    events = run(service([hit(1)]))
    assert [n["kind"] for n in notices(events)] == ["web_search_off"]
    assert events[-1][0] == "done"


def test_search_failure_is_a_notice_and_the_answer_continues():
    search = FakeSearch(error=SearchError("web_search_failed", "DuckDuckGo returned 202"))
    events = run(service([hit(1)], normal_search=search))
    assert notices(events) == [{"kind": "web_search_failed", "message": "DuckDuckGo returned 202"}]
    assert ("web", []) in events
    assert events[-1][0] == "done"


def test_unexpected_search_exception_is_a_notice_without_internal_details():
    events = run(service([hit(1)], normal_search=FakeSearch(error=RuntimeError("parser broke"))))
    assert notices(events) == [{"kind": "web_search_failed", "message": "Web search failed"}]
    assert events[-1][0] == "done"


def test_blocked_search_is_an_egress_notice():
    search = FakeSearch(error=SearchError("egress_blocked", "The egress guard blocked DuckDuckGo"))
    events = run(service([hit(1)], normal_search=search))
    assert [n["kind"] for n in notices(events)] == ["egress_blocked"]


def test_private_mode_uses_only_the_private_search():
    normal, private = FakeSearch([WEB]), FakeSearch([])
    run(
        service([hit(1)], models={"ollama": FakeModel()}, normal_search=normal, private_search=private),
        model="ollama",
        private=True,
    )
    assert normal.queries == []
    assert private.queries == [QUESTION]


def test_private_mode_without_a_private_search_does_not_use_the_normal_one():
    normal = FakeSearch([WEB])
    events = run(
        service([hit(1)], models={"ollama": FakeModel()}, normal_search=normal), model="ollama", private=True
    )
    assert normal.queries == []
    assert [n["kind"] for n in notices(events)] == ["web_search_off"]


def test_private_mode_answers_with_ollama_even_when_gemini_was_requested():
    gemini, ollama = FakeModel(pieces=("G",)), FakeModel(pieces=("O",))
    events = run(service([hit(1)], models={"gemini": gemini, "ollama": ollama}), model="gemini", private=True)
    assert gemini.calls == []
    assert len(ollama.calls) == 1
    assert "private_mode_model" in [n["kind"] for n in notices(events)]
    assert events[-1][0] == "done"


def test_private_mode_without_ollama_fails_and_never_uses_gemini():
    gemini = FakeModel(pieces=("G",))
    events = run(service([hit(1)], models={"gemini": gemini}), model="gemini", private=True)
    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "ollama_unavailable"
    assert gemini.calls == []


def test_private_mode_ollama_failure_is_not_retried_with_gemini():
    gemini = FakeModel(pieces=("G",))
    ollama = FakeModel(error=ProviderError("ollama_unavailable", "Ollama is not reachable"))
    events = run(service([hit(1)], models={"gemini": gemini, "ollama": ollama}), model="ollama", private=True)
    assert events[-1] == ("error", {"code": "ollama_unavailable", "message": "Ollama is not reachable"})
    assert gemini.calls == []


def test_provider_error_ends_with_error_event():
    model = FakeModel(error=ProviderError("ollama_unavailable", "Ollama is not reachable"))
    events = run(service([hit(1)], model=model))
    assert events[-1] == ("error", {"code": "ollama_unavailable", "message": "Ollama is not reachable"})
    assert "done" not in names(events)


def test_unknown_model_is_an_error():
    events = run(service([hit(1)]), model="gemini")
    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "model_unavailable"


def test_detection_is_appended_to_history():
    history = []
    run(service([hit(1)]), history=history)
    assert [d.chosen for d in history] == ["ind_Latn"]


def test_finished_answer_is_recorded_with_invalid_citations_removed():
    records = []
    run(service([hit(1)], normal_search=FakeSearch([WEB])), record=records.append)
    [exchange] = records
    assert exchange.question == QUESTION
    assert exchange.answer == "Jawaban [1]"
    assert exchange.answer_language == "ind_Latn"
    assert exchange.detection["chosen"] == "ind_Latn"
    assert exchange.sources[0]["title"] == "T1"
    assert exchange.web == [WEB_PAYLOAD]
    assert exchange.citations == {"valid": [1], "removed": [5]}


def test_failed_answer_is_not_recorded():
    records = []
    model = FakeModel(error=ProviderError("provider_error", "boom"))
    run(service([hit(1)], model=model), record=records.append)
    assert records == []
