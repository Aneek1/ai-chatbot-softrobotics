from backend.pipeline.langid import Candidate, TwoStageDetector
from backend.pipeline.languages import CONFUSION_GROUPS
from tests.fakes import FakeScorer


def detector(candidates, specialist=None, min_confidence=0.6):
    return TwoStageDetector(
        general=FakeScorer(candidates), min_confidence=min_confidence, specialist=specialist
    )


def test_confident_general_result():
    result = detector([("jpn_Jpan", 0.99), ("zho_Hans", 0.01)]).detect("ソフトロボット")
    assert result.chosen == "jpn_Jpan"
    assert result.uncertain is False
    assert result.stage == "general"
    assert result.candidates[0] == Candidate("jpn_Jpan", 0.99)


def test_low_confidence_is_uncertain():
    result = detector([("ind_Latn", 0.41), ("zsm_Latn", 0.39)]).detect("silikon")
    assert result.uncertain is True


def test_han_label_resolved_to_simplified_by_rule():
    result = detector([("cmn_Hani", 1.0)]).detect("如何制作软气动执行器？")
    assert result.chosen == "zho_Hans"
    assert result.stage == "rule"
    assert result.uncertain is False


def test_han_label_with_ambiguous_characters_is_uncertain():
    result = detector([("cmn_Hani", 0.99)]).detect("中文")
    assert result.chosen == "cmn_Hani"
    assert result.uncertain is True


def test_specialist_decides_inside_a_confusion_group():
    specialist = FakeScorer([("zsm_Latn", 0.93), ("ind_Latn", 0.07)])
    result = detector([("ind_Latn", 0.72), ("zsm_Latn", 0.23)], specialist=specialist).detect(
        "Bagaimanakah cara untuk membuat penggerak pneumatik lembut?"
    )
    assert result.chosen == "zsm_Latn"
    assert result.stage == "specialist"


def test_specialist_not_called_outside_groups():
    specialist = FakeScorer([], fail_if_called=True)
    result = detector([("kor_Hang", 1.0)], specialist=specialist).detect("소프트 로봇")
    assert result.chosen == "kor_Hang"


def test_mixed_script_text_is_uncertain_even_when_confident():
    result = detector([("eng_Latn", 0.95)]).detect("软体机器人 soft robot")
    assert result.uncertain is True


def test_newlines_are_removed_before_scoring():
    scorer = FakeScorer([("eng_Latn", 0.99)])
    TwoStageDetector(general=scorer, min_confidence=0.6).detect("line one\nline two")
    assert scorer.seen == ["line one line two"]


def test_empty_text_is_undetermined():
    result = detector([("eng_Latn", 0.99)]).detect("   \n ")
    assert result.chosen == "und"
    assert result.uncertain is True
    assert result.candidates == ()


def test_confusion_groups_are_disjoint():
    # group_of() returns the first matching group, so a label in two groups would route silently.
    labels = [label for members in CONFUSION_GROUPS.values() for label in members]
    assert len(labels) == len(set(labels))


def test_specialist_labels_outside_the_group_are_ignored():
    # A specialist trained only for Malay/Indonesian must not relabel Chinese text.
    specialist = FakeScorer([("ind_Latn", 0.8), ("zsm_Latn", 0.2)])
    result = detector([("cmn_Hani", 0.9)], specialist=specialist).detect("如何制作软气动执行器？")
    assert result.chosen == "zho_Hans"
    assert result.stage == "rule"


def test_empty_general_result_is_undetermined():
    result = detector([]).detect("hello")
    assert result.chosen == "und"
    assert result.uncertain is True


def test_specialist_without_group_labels_keeps_the_general_result():
    result = detector([("ind_Latn", 0.95)], specialist=FakeScorer([])).detect("Bagaimana cara membuat aktuator?")
    assert result.chosen == "ind_Latn"
    assert result.stage == "general"


def test_han_rule_does_not_duplicate_the_chosen_label():
    result = detector([("cmn_Hani", 0.9), ("zho_Hans", 0.05)]).detect("如何制作软气动执行器？")
    assert [c.code for c in result.candidates].count("zho_Hans") == 1
