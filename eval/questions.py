"""Evaluation questions (spec section 11.2).

Twenty questions are written by hand in English, one file per set, and machine-translated into every
supported language by eval/translate_questions.py. Each question names a topic; eval/topics.py says
which documents answer it. No question has been checked by a native speaker, so
verified_by_native_speaker is false everywhere and results are not split by it yet.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from backend.pipeline.languages import SUPPORTED
from eval.topics import load_topics

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGLISH_PATH = REPO_ROOT / "eval" / "questions-en.jsonl"
QUESTIONS_PATH = REPO_ROOT / "eval" / "questions.jsonl"
FIELDS = ("id", "language", "topic", "text", "translated_by", "verified_by_native_speaker")


class QuestionError(ValueError):
    """A question file the evaluation cannot use."""


@dataclass(frozen=True)
class Question:
    id: str
    language: str
    topic: str
    text: str
    translated_by: str  # empty when the question was written by hand
    verified_by_native_speaker: bool


def validate(questions: Sequence[Question], topics: dict[str, list[str]]) -> None:
    seen: set[str] = set()
    for question in questions:
        if question.language not in SUPPORTED:
            raise QuestionError(f"{question.id}: unsupported language {question.language!r}")
        if question.topic not in topics:
            raise QuestionError(f"{question.id}: unknown topic {question.topic!r}")
        if not question.text.strip():
            raise QuestionError(f"{question.id}: empty text")
        if question.id in seen:
            raise QuestionError(f"duplicate question id {question.id!r}")
        seen.add(question.id)


def read_questions(path: Path, topics: dict[str, list[str]] | None = None) -> list[Question]:
    questions = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = [name for name in FIELDS if name not in record]
        if missing:
            raise QuestionError(f"{path.name} line {number}: missing {', '.join(missing)}")
        questions.append(Question(**{name: record[name] for name in FIELDS}))
    validate(questions, load_topics() if topics is None else topics)
    return questions


def write_questions(questions: Sequence[Question], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for question in questions:
            handle.write(json.dumps(asdict(question), ensure_ascii=False, sort_keys=True) + "\n")
    return len(questions)
