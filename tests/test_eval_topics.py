import json

import pytest

from eval.topics import TopicError, load_topics, missing_from


def test_the_committed_topics_load():
    topics = load_topics()
    assert set(topics) == {
        "3D printing",
        "Elastomer",
        "Pneumatic actuator",
        "Polydimethylsiloxane",
        "Silicone rubber",
        "Soft robotics",
    }
    assert "wikipedia:en:50647426" in topics["Soft robotics"]
    assert sum(len(ids) for ids in topics.values()) == 32


def test_a_topic_without_documents_is_refused(tmp_path):
    path = tmp_path / "topics.json"
    path.write_text(json.dumps({"Soft robotics": []}), encoding="utf-8")
    with pytest.raises(TopicError, match="no document ids"):
        load_topics(path)


def test_only_wikipedia_ids_are_allowed(tmp_path):
    path = tmp_path / "topics.json"
    path.write_text(json.dumps({"Soft robotics": ["arxiv:2609.00751"]}), encoding="utf-8")
    with pytest.raises(TopicError, match="not a Wikipedia document id"):
        load_topics(path)


def test_missing_from_lists_only_what_the_index_lacks():
    topics = {"A": ["wikipedia:en:1", "wikipedia:id:2"], "B": ["wikipedia:en:3"]}
    assert missing_from(topics, {"wikipedia:en:1", "wikipedia:en:3"}) == {"A": ["wikipedia:id:2"]}
