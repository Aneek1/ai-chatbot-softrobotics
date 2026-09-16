from collections.abc import Sequence

from backend.pipeline.langid import Detection

DEFAULT_LANGUAGE = "eng_Latn"


def choose_answer_language(detection: Detection, override: str | None, history: Sequence[Detection]) -> str:
    if override:
        return override
    if not detection.uncertain:
        return detection.chosen
    for past in reversed(history):
        if not past.uncertain:
            return past.chosen
    return DEFAULT_LANGUAGE
