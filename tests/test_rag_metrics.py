from eval.rag_metrics import METRICS, cited_doc_ids, hit_at_k, rate, summarize


def record(language, topic="Soft robotics", **overrides):
    values = dict.fromkeys(METRICS, True)
    return {"language": language, "topic": topic, **values, **overrides}


def test_hit_at_k_looks_only_at_the_first_k():
    retrieved = ["a", "b", "c", "d", "e", "f"]
    assert hit_at_k(retrieved, ["f"], 6) is True
    assert hit_at_k(retrieved, ["f"], 5) is False
    assert hit_at_k(retrieved, [], 6) is False


def test_hit_at_k_accepts_any_expected_document():
    assert hit_at_k(["b"], ["a", "b"], 5) is True


def test_cited_doc_ids_maps_numbers_to_documents_and_drops_out_of_range():
    assert cited_doc_ids([1, 3, 9], ["d1", "d2", "d3"]) == ["d1", "d3"]


def test_rate_of_nothing_is_none():
    assert rate([]) is None
    assert rate([True, False, True, True]) == 0.75


def test_summarize_groups_by_language():
    records = [record("eng_Latn"), record("eng_Latn", hit_at_5=False), record("tam_Taml")]
    summary = summarize(records)
    assert summary["eng_Latn"]["questions"] == 2
    assert summary["eng_Latn"]["hit_at_5"] == 0.5
    assert summary["eng_Latn"]["answer_language_correct"] == 1.0
    assert summary["tam_Taml"]["questions"] == 1


def test_summarize_can_group_by_topic_and_ignores_missing_values():
    records = [record("eng_Latn", topic="Elastomer", citations_all_valid=None)]
    summary = summarize(records, key="topic")
    assert list(summary) == ["Elastomer"]
    assert summary["Elastomer"]["citations_all_valid"] is None
