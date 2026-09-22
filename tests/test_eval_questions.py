import json

import pytest

from eval.questions import (
    ENGLISH_PATH,
    Question,
    QuestionError,
    read_questions,
    validate,
    write_questions,
)
from eval.topics import load_topics

TOPICS = {"Soft robotics": ["wikipedia:en:50647426"]}


def question(**overrides) -> Question:
    values = {
        "id": "q01-eng_Latn",
        "language": "eng_Latn",
        "topic": "Soft robotics",
        "text": "What is soft robotics?",
        "translated_by": "",
        "verified_by_native_speaker": False,
    }
    return Question(**(values | overrides))


def test_the_english_questions_load_and_cover_every_topic():
    questions = read_questions(ENGLISH_PATH)
    topics = load_topics()
    assert len(questions) == 20
    assert {q.language for q in questions} == {"eng_Latn"}
    assert {q.topic for q in questions} == set(topics)
    assert all(q.translated_by == "" for q in questions)
    assert all(q.verified_by_native_speaker is False for q in questions)


def test_an_unknown_topic_is_refused():
    with pytest.raises(QuestionError, match="unknown topic"):
        validate([question(topic="Hydraulics")], TOPICS)


def test_an_unsupported_language_is_refused():
    with pytest.raises(QuestionError, match="unsupported language"):
        validate([question(language="fra_Latn")], TOPICS)


def test_duplicate_ids_are_refused():
    with pytest.raises(QuestionError, match="duplicate question id"):
        validate([question(), question()], TOPICS)


def test_empty_text_is_refused():
    with pytest.raises(QuestionError, match="empty text"):
        validate([question(text="   ")], TOPICS)


def test_write_then_read_round_trips(tmp_path):
    path = tmp_path / "questions.jsonl"
    written = [question(), question(id="q02-zho_Hant", language="zho_Hant", text="什麼是軟體機器人？")]
    assert write_questions(written, path) == 2
    assert read_questions(path, TOPICS) == written
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[1])["language"] == "zho_Hant"
