import pytest

from eval.questions import Question
from eval.translate_questions import (
    TARGETS,
    TranslationError,
    clean_translation,
    translate,
    translate_all,
    translation_prompt,
)
from tests.fakes import FakeModel

ENGLISH = Question(
    id="q01",
    language="eng_Latn",
    topic="Soft robotics",
    text="What is soft robotics?",
    translated_by="",
    verified_by_native_speaker=False,
)


def test_the_prompt_names_the_target_language_and_asks_for_the_translation_only():
    system, user = translation_prompt(ENGLISH, "hin_Latn")
    assert "Hindi (romanized)" in system.content
    assert "translation only" in system.content
    assert user.content == "What is soft robotics?"


def test_english_passes_through_untranslated():
    model = FakeModel(pieces=("should not be used",))
    result = translate(ENGLISH, "eng_Latn", model, "test-model")
    assert result.id == "q01-eng_Latn"
    assert result.text == "What is soft robotics?"
    assert result.translated_by == ""
    assert model.calls == []


def test_a_translation_is_cleaned_and_attributed():
    model = FakeModel(pieces=('"什麼是', '軟體機器人？"\n'))
    result = translate(ENGLISH, "zho_Hant", model, "test-model")
    assert result == Question(
        id="q01-zho_Hant",
        language="zho_Hant",
        topic="Soft robotics",
        text="什麼是軟體機器人?",
        translated_by="test-model",
        verified_by_native_speaker=False,
    )


def test_an_empty_translation_stops_the_run():
    with pytest.raises(TranslationError, match="q01"):
        translate(ENGLISH, "tam_Taml", FakeModel(pieces=('""',)), "test-model")


def test_translate_all_covers_every_supported_language():
    results = translate_all([ENGLISH], FakeModel(pieces=("terjemahan",)), "test-model")
    assert len(results) == len(TARGETS) == 10
    assert len({q.id for q in results}) == 10
    assert clean_translation(" x \n") == "x"
