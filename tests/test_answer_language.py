from backend.pipeline.answer_language import choose_answer_language
from backend.pipeline.langid import Detection


def det(chosen, uncertain):
    return Detection(candidates=(), chosen=chosen, uncertain=uncertain, stage="general")


def test_override_wins():
    assert choose_answer_language(det("ind_Latn", False), "tam_Taml", []) == "tam_Taml"


def test_confident_detection_is_used():
    assert choose_answer_language(det("zsm_Latn", False), None, []) == "zsm_Latn"


def test_uncertain_falls_back_to_last_confident_message():
    history = [det("kor_Hang", False), det("und", True)]
    assert choose_answer_language(det("ind_Latn", True), None, history) == "kor_Hang"


def test_english_when_nothing_is_confident():
    assert choose_answer_language(det("und", True), None, [det("und", True)]) == "eng_Latn"
