import random

import pytest

from eval.judge_agreement import (
    SampleError,
    agreement,
    check_no_labels_to_lose,
    draft_sample,
    read_sample,
    write_sample,
)


def judged(n, verdict):
    return [
        {
            "question_id": f"q{i}-eng_Latn",
            "language": "eng_Latn",
            "citation": 1,
            "chunk_id": f"d{i}:0",
            "sentence": f"Sentence {i} [1].",
            "chunk_text": f"Passage {i}.",
            "verdict": verdict,
        }
        for i in range(n)
    ]


def test_the_draft_is_stratified_over_the_verdicts_and_left_unlabelled():
    results = {"judged": judged(10, "supported") + judged(4, "unsupported") + judged(1, "unclear")}
    sample = draft_sample(results, count=9, rng=random.Random(0))
    assert len(sample) == 7  # 3 supported, 3 unsupported, 1 unclear
    assert sorted(row["judge"] for row in sample) == ["supported"] * 3 + ["unclear"] + ["unsupported"] * 3
    assert {row["human"] for row in sample} == {""}
    assert all(row["sentence"] and row["chunk_text"] for row in sample)


def test_agreement_counts_only_labelled_rows():
    sample = [
        {"judge": "supported", "human": "supported"},
        {"judge": "supported", "human": "unsupported"},
        {"judge": "unsupported", "human": "unsupported"},
        {"judge": "unclear", "human": ""},
    ]
    assert agreement(sample) == {
        "pairs": 4,
        "labelled": 3,
        "agreement": 0.6667,
        "judge_supported_rate": 0.6667,
        "human_supported_rate": 0.3333,
    }


def test_an_unknown_label_is_refused():
    with pytest.raises(SampleError, match="yes"):
        agreement([{"judge": "supported", "human": "yes"}])


def test_write_then_read_round_trips(tmp_path):
    path = tmp_path / "sample.jsonl"
    rows = [{"judge": "supported", "human": "", "sentence": "A [1].", "chunk_text": "A."}]
    write_sample(rows, path)
    assert read_sample(path) == rows


def test_drafting_over_a_labelled_sample_is_refused_unless_forced(tmp_path):
    path = tmp_path / "sample.jsonl"
    assert check_no_labels_to_lose(path, force=False) is None  # a missing file is fine
    write_sample([{"judge": "supported", "human": ""}], path)
    assert check_no_labels_to_lose(path, force=False) is None  # an unlabelled draft is fine
    write_sample([{"judge": "supported", "human": "supported"}], path)
    with pytest.raises(SystemExit, match="already has labelled rows"):
        check_no_labels_to_lose(path, force=False)
    assert check_no_labels_to_lose(path, force=True) is None
