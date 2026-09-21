"""Translate the English evaluation questions into every supported language (spec section 11.2).

Usage:
    PYTHONUTF8=1 OLLAMA_MODEL=qwen3:30b-a3b-instruct-2507-q4_K_M \\
        uv run python -m eval.translate_questions

The translations are a model's work. Nothing here was checked by a native speaker, so every
translated question is written with verified_by_native_speaker false, and the results file names the
model that produced them.
"""

from dataclasses import replace
from pathlib import Path

from backend.app.config import Settings
from backend.pipeline.languages import SUPPORTED, display_name
from backend.pipeline.normalize import normalize
from backend.providers.base import AnswerModel, Message
from backend.providers.ollama import OllamaModel
from eval.questions import ENGLISH_PATH, QUESTIONS_PATH, Question, read_questions, write_questions

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGETS = tuple(SUPPORTED)
QUOTES = '"“”「」«»'


class TranslationError(RuntimeError):
    """A translation the evaluation cannot use."""


def translation_prompt(question: Question, language: str) -> list[Message]:
    system = (
        "You translate short questions about soft-robotics fabrication. "
        f"Translate the question into {display_name(language)}. "
        "Keep it one question, keep the meaning, and answer with the translation only: "
        "no quotation marks, no explanation, no alternatives."
    )
    return [Message("system", system), Message("user", question.text)]


def clean_translation(text: str) -> str:
    return normalize(text).strip(QUOTES).strip()


def translate(question: Question, language: str, model: AnswerModel, model_name: str) -> Question:
    if language == question.language:
        return replace(question, id=f"{question.id}-{language}")
    text = clean_translation("".join(model.stream(translation_prompt(question, language))))
    if not text:
        raise TranslationError(f"{question.id}: {model_name} returned no translation into {language}")
    return Question(
        id=f"{question.id}-{language}",
        language=language,
        topic=question.topic,
        text=text,
        translated_by=model_name,
        verified_by_native_speaker=False,
    )


def translate_all(
    questions: list[Question],
    model: AnswerModel,
    model_name: str,
    languages: tuple[str, ...] = TARGETS,
) -> list[Question]:
    return [
        translate(question, language, model, model_name)
        for language in languages
        for question in questions
    ]


def main() -> None:
    settings = Settings()
    model = OllamaModel(settings.ollama_url, settings.ollama_model, settings.ollama_timeout)
    english = read_questions(ENGLISH_PATH)
    translated = translate_all(english, model, settings.ollama_model)
    written = write_questions(translated, QUESTIONS_PATH)
    print(f"Wrote {written} questions to {QUESTIONS_PATH.relative_to(REPO_ROOT)}")
    for language in TARGETS:
        example = next(q for q in translated if q.language == language)
        print(f"{language}: {example.text}")


if __name__ == "__main__":
    main()
