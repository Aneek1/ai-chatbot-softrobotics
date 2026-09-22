import pytest

from backend.pipeline.langid import TwoStageDetector
from backend.pipeline.retrieve import Chunk, Hit
from eval.questions import Question
from eval.rag_eval import ContaminatedIndex, answer_question, evaluate
from tests.fakes import FakeModel, FakeScorer

TOPICS = {"Silicone rubber": ["wikipedia:en:3342099"]}


def question(language="eng_Latn", topic="Silicone rubber"):
    return Question(
        id=f"q05-{language}",
        language=language,
        topic=topic,
        text="What is silicone rubber made of?",
        translated_by="",
        verified_by_native_speaker=False,
    )


def hit(doc_id, text="Silicone rubber is a polymer with silicon."):
    chunk = Chunk(
        f"{doc_id}:0", doc_id, text, "eng_Latn", "wikipedia", "Silicone rubber",
        "https://example.invalid", "CC BY-SA 4.0",
    )
    return Hit(chunk, 0.8)


class FakeRetriever:
    def __init__(self, hits):
        self._hits = hits
        self.queries = []

    def search(self, query, top_k):
        self.queries.append(query)
        return self._hits[:top_k]


def detector(code="eng_Latn"):
    return TwoStageDetector(general=FakeScorer([(code, 0.95)]), min_confidence=0.6)


def test_a_question_is_answered_scored_and_paired():
    retriever = FakeRetriever([hit("wikipedia:en:3342099"), hit("arxiv:1")])
    model = FakeModel(pieces=("Silicone rubber contains silicon [1].", " Invented in 1806 [7]."))
    record, hits, answer, pairs = answer_question(question(), detector(), retriever, model, TOPICS, 6)
    assert record["hit_at_5"] is True and record["hit_at_6"] is True
    assert record["answer_language_detected"] == "eng_Latn"
    assert record["answer_language_correct"] is True
    assert record["has_citation"] is True
    assert record["citations_all_valid"] is False
    assert record["citations_removed"] == [7]
    assert record["retrieved_doc_ids"] == ["wikipedia:en:3342099", "arxiv:1"]
    assert "[7]" not in answer
    assert [p.citation for p in pairs] == [1]
    assert len(hits) == 2


def test_a_missed_topic_and_a_wrong_answer_language_are_recorded():
    retriever = FakeRetriever([hit("arxiv:1")])
    record, _, _, pairs = answer_question(
        question("tam_Taml"), detector("eng_Latn"), retriever, FakeModel(pieces=("No citation.",)),
        TOPICS, 6,
    )
    assert record["hit_at_5"] is False
    assert record["answer_language_correct"] is False
    assert record["has_citation"] is False
    assert record["citations_all_valid"] is None
    assert pairs == []


def test_evaluate_summarizes_per_language_and_per_topic():
    retriever = FakeRetriever([hit("wikipedia:en:3342099")])
    results = evaluate(
        [question(), question("ind_Latn")],
        detector(),
        retriever,
        FakeModel(pieces=("Silicone contains silicon [1].",)),
        TOPICS,
        top_k=6,
        log=lambda message: None,
    )
    assert results["questions"] == 2
    assert sorted(results["per_language"]) == ["eng_Latn", "ind_Latn"]
    assert results["per_language"]["eng_Latn"]["hit_at_5"] == 1.0
    assert results["overall"]["questions"] == 2
    assert list(results["per_topic"]) == ["Silicone rubber"]
    assert results["faithfulness"] is None and results["judged"] == []
    assert retriever.queries == ["What is silicone rubber made of?"] * 2


def test_evaluate_judges_the_pairs_when_a_judge_is_given():
    class YesJudge:
        name = "yes-judge"

        def verdict(self, pair):
            return "supported"

    results = evaluate(
        [question()],
        detector(),
        FakeRetriever([hit("wikipedia:en:3342099")]),
        FakeModel(pieces=("Silicone contains silicon [1].",)),
        TOPICS,
        top_k=6,
        judge=YesJudge(),
        log=lambda message: None,
    )
    assert results["faithfulness"]["supported_rate"] == 1.0
    assert results["faithfulness_by_language"]["eng_Latn"]["pairs"] == 1
    assert results["judge"] == "yes-judge"


def test_a_test_only_chunk_stops_the_run():
    """A test fixture indexed into the real index must never answer a real question."""
    fixture = Hit(
        Chunk(
            "fixture-en:0", "fixture-en",
            "Fixture text about silicone moulding. It exists only for tests.",
            "eng_Latn", "fixture", "Silicone fixture", "https://example.invalid/en", "test-only",
        ),
        0.9,
    )
    retriever = FakeRetriever([hit("wikipedia:en:3342099"), fixture])
    with pytest.raises(ContaminatedIndex, match="fixture-en"):
        answer_question(question(), detector(), retriever, FakeModel(pieces=("a",)), TOPICS, 6)
