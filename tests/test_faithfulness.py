from backend.pipeline.retrieve import Chunk, Hit
from eval.faithfulness import (
    ModelJudge,
    Pair,
    faithfulness,
    judge_all,
    judge_prompt,
    pairs_for,
    parse_verdict,
    sentences_with_citations,
)
from eval.questions import Question
from tests.fakes import FakeModel

QUESTION = Question(
    id="q05-eng_Latn",
    language="eng_Latn",
    topic="Silicone rubber",
    text="What is silicone rubber made of?",
    translated_by="",
    verified_by_native_speaker=False,
)


def hit(n, text):
    chunk = Chunk(
        f"d{n}:0", f"d{n}", text, "eng_Latn", "wikipedia", f"T{n}", "https://example.invalid", "CC BY-SA 4.0"
    )
    return Hit(chunk, 0.8)


def test_sentences_keep_their_citation_numbers():
    assert sentences_with_citations("Silicone cures [1]. It is elastic [2][3].", "Latn") == [
        ("Silicone cures [1].", [1]),
        ("It is elastic [2][3].", [2, 3]),
    ]


def test_cjk_answers_split_on_their_own_punctuation():
    pairs = sentences_with_citations("矽膠會硫化[1]。它有彈性[2]。", "Hani")
    assert [numbers for _, numbers in pairs] == [[1], [2]]


def test_pairs_carry_the_question_language_and_the_cited_chunk():
    pairs = pairs_for(QUESTION, "Silicone cures [1]. Nothing here.", [hit(1, "Silicone cures at 150 C.")])
    assert pairs == [
        Pair(
            question_id="q05-eng_Latn",
            language="eng_Latn",
            sentence="Silicone cures [1].",
            citation=1,
            chunk_id="d1:0",
            chunk_text="Silicone cures at 150 C.",
        )
    ]


def test_out_of_range_and_repeated_citations_are_dropped():
    pairs = pairs_for(QUESTION, "A [1][1][9].", [hit(1, "text")])
    assert [(p.citation, p.chunk_id) for p in pairs] == [(1, "d1:0")]


def test_the_judge_prompt_shows_the_passage_and_the_sentence():
    pair = pairs_for(QUESTION, "A [1].", [hit(1, "passage text")])[0]
    system, user = judge_prompt(pair)
    assert "supported" in system.content and "unsupported" in system.content
    assert "passage text" in user.content and "A [1]." in user.content


def test_verdicts_are_parsed_without_unsupported_reading_as_supported():
    assert parse_verdict("Unsupported.") == "unsupported"
    assert parse_verdict(" Supported ") == "supported"
    assert parse_verdict("I cannot tell") == "unclear"


def test_faithfulness_counts_every_verdict():
    pair = pairs_for(QUESTION, "A [1]. B [1].", [hit(1, "text")])
    judged = judge_all(pair, ModelJudge(FakeModel(pieces=("supported",)), "test-model"))
    assert [j["verdict"] for j in judged] == ["supported", "supported"]
    assert faithfulness(judged) == {
        "pairs": 2,
        "supported": 2,
        "unsupported": 0,
        "unclear": 0,
        "supported_rate": 1.0,
    }
    assert faithfulness([])["supported_rate"] is None
