import pytest

from eval.flores_benchmark import (
    PUBLISHED,
    contamination_check,
    model_labels,
    native_space,
    scorer_codes,
    specialist_space,
)
from tests.fakes import FakeScorer
from training.data.rows import Row, write_jsonl


def test_native_space_merges_chinese_and_excludes_codes_the_model_lacks():
    devtest = {"eng_Latn": ["x", "y"], "est_Latn": ["e"], "zho_Hans": ["s"], "zho_Hant": ["t"]}
    result = native_space(
        lambda texts: ["eng_Latn", "cmn_Hani", "cmn_Hani", "cmn_Hani"],
        devtest,
        {"eng_Latn", "cmn_Hani", "ekk_Latn"},
    )
    assert result["languages"] == 2 and result["sentences"] == 4
    assert result["excluded_flores_codes"] == ["est_Latn"]
    assert result["per_label_f1"] == {"cmn_Hani": 0.8, "eng_Latn": 0.6667}
    assert result["macro_f1"] == pytest.approx(0.7333, abs=1e-4)
    assert result["macro_fpr"] == 0.25


class AllLabelsScorer:
    """fastText returns every label for k=-1; FakeScorer slices, so it cannot stand in here."""

    def __init__(self):
        self.seen = []

    def top(self, text, k):
        self.seen.append(text)
        candidates = FakeScorer([("eng_Latn", 0.7), ("sco_Latn", 0.2), ("fra_Latn", 0.1)]).top(text, 3)
        return candidates if k == -1 else candidates[:k]


def test_scorer_codes_and_labels_come_from_the_model():
    scorer = AllLabelsScorer()
    assert scorer_codes(scorer)(["hello\nthere"]) == ["eng_Latn"]
    assert scorer.seen == ["hello there"]
    assert model_labels(scorer) == {"eng_Latn", "sco_Latn", "fra_Latn"}


def test_specialist_space_counts_unsupported_languages_as_other():
    devtest = {"eng_Latn": ["hello there"], "fra_Latn": ["bonjour"], "zsm_Latn": ["apa khabar"]}

    class Fixed:
        name = "fixed"

        def predict_many(self, texts):
            return [("eng_Latn", 0.9), ("other", 0.8), ("ind_Latn", 0.7)]

    result = specialist_space(Fixed(), devtest)
    assert result["count"] == 3
    assert result["per_label"]["other"]["f1"] == 1.0
    assert result["confusions"]["malay_indonesian"]["zsm_Latn"]["ind_Latn"] == 1
    assert set(result["false_positive_rate"]) == {"eng_Latn", "zsm_Latn", "other"}


def test_contamination_check_counts_matching_rows(tmp_path):
    write_jsonl(
        [Row("The FLORES  sentence.", "eng_Latn", "openlid", "a"), Row("clean", "eng_Latn", "openlid", "a")],
        tmp_path / "train.jsonl",
    )
    write_jsonl([Row("clean too", "eng_Latn", "openlid", "b")], tmp_path / "test.jsonl")
    result = contamination_check(
        tmp_path, {"flores200-devtest": ["The FLORES sentence."], "dakshina-test": ["kitab"]}
    )
    assert result == {"flores200-devtest": {"train": 1, "test": 0}, "dakshina-test": {"train": 0, "test": 0}}


def test_published_figures_are_devtest_and_cite_a_table():
    assert {p["split"] for p in PUBLISHED} == {"devtest"}
    assert all("Table" in p["source"] and "arXiv:" in p["source"] for p in PUBLISHED)
    table_3 = [p for p in PUBLISHED if "Table 3" in p["source"]]
    assert len(table_3) == 2
    assert all('"SET?": language set not known to the model' in p["source"] for p in table_3)
    glotlid_all = PUBLISHED[0]
    assert (glotlid_all["languages"], glotlid_all["macro_f1"], glotlid_all["macro_fpr"]) == (
        196,
        0.917,
        0.0042,
    )
