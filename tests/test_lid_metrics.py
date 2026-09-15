import pytest

from training.metrics import (
    OUTSIDE,
    confusion_matrix,
    false_positive_rates,
    macro_f1,
    per_label_scores,
    report,
)

GOLD = ["ind_Latn", "ind_Latn", "zsm_Latn", "zsm_Latn", "eng_Latn"]
PRED = ["ind_Latn", "zsm_Latn", "zsm_Latn", "ind_Latn", "eng_Latn"]


def test_per_label_scores_by_hand():
    scores = per_label_scores(GOLD, PRED, ["ind_Latn", "zsm_Latn", "eng_Latn"])
    assert scores["ind_Latn"].precision == 0.5
    assert scores["ind_Latn"].recall == 0.5
    assert scores["ind_Latn"].f1 == 0.5
    assert scores["eng_Latn"].f1 == 1.0
    assert scores["zsm_Latn"].support == 2


def test_labels_absent_from_gold_do_not_lower_macro_f1():
    scores = per_label_scores(GOLD, PRED, ["ind_Latn", "zsm_Latn", "eng_Latn", "kor_Hang"])
    assert scores["kor_Hang"].support == 0
    assert macro_f1(scores) == pytest.approx((0.5 + 0.5 + 1.0) / 3)


def test_length_mismatch_is_an_error():
    with pytest.raises(ValueError, match="gold has 1 items but pred has 2"):
        per_label_scores(["a"], ["a", "b"], ["a"])


def test_confusion_matrix_collapses_outside_predictions():
    matrix = confusion_matrix(
        ["ind_Latn", "zsm_Latn", "zsm_Latn"], ["ind_Latn", "other", "ind_Latn"], ["ind_Latn", "zsm_Latn"]
    )
    assert matrix["zsm_Latn"] == {"ind_Latn": 1, "zsm_Latn": 0, OUTSIDE: 1}
    assert matrix["ind_Latn"]["ind_Latn"] == 1


def test_false_positive_rates():
    rates = false_positive_rates(GOLD, PRED, ["ind_Latn", "eng_Latn"])
    # One zsm_Latn line predicted as ind_Latn, out of three lines whose gold is not ind_Latn.
    assert rates["ind_Latn"] == pytest.approx(1 / 3)
    assert rates["eng_Latn"] == 0.0


def test_report_has_group_scores_and_confusions():
    groups = {"malay_indonesian": ("ind_Latn", "zsm_Latn")}
    result = report(GOLD, PRED, ["ind_Latn", "zsm_Latn", "eng_Latn"], groups)
    assert result["count"] == 5
    assert result["accuracy"] == 0.6
    assert result["group_macro_f1"] == 0.5
    assert result["group_f1"] == {"malay_indonesian": 0.5}
    assert result["macro_f1"] == 0.6667
    assert result["confusions"]["malay_indonesian"]["ind_Latn"]["zsm_Latn"] == 1
    assert result["per_label"]["eng_Latn"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1}
