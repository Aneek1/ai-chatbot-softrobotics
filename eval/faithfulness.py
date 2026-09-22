"""Does the cited chunk support the sentence citing it (spec section 11.2)?

The app's check_citations only checks that [n] points at a source that exists. In Plan 1's
end-to-end run the model cited fixture chunks for details those chunks do not contain, and that
check called them valid. This pass reads the sentence together with the chunk it cites and asks a
model whether the chunk states what the sentence claims.

The verdicts are a model's judgement. eval/judge_agreement.py measures how often that judge agrees
with a hand-labelled sample, and every reported faithfulness number is published next to it.
"""

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from backend.pipeline.chunk import split_sentences
from backend.pipeline.normalize import script_profile
from backend.pipeline.retrieve import Hit
from backend.providers.base import AnswerModel, Message
from eval.questions import Question

CITATION = re.compile(r"\[(\d+)\]")
VERDICTS = ("supported", "unsupported", "unclear")


@dataclass(frozen=True)
class Pair:
    question_id: str
    language: str
    sentence: str
    citation: int
    chunk_id: str
    chunk_text: str


def sentences_with_citations(answer: str, script: str | None) -> list[tuple[str, list[int]]]:
    return [
        (sentence, [int(number) for number in CITATION.findall(sentence)])
        for sentence in split_sentences(answer, script)
    ]


def pairs_for(question: Question, answer: str, hits: Sequence[Hit]) -> list[Pair]:
    script = script_profile(answer).dominant
    pairs = []
    for sentence, numbers in sentences_with_citations(answer, script):
        # dict.fromkeys keeps the first occurrence of each number and drops the repeats.
        for number in dict.fromkeys(numbers):
            if 1 <= number <= len(hits):
                chunk = hits[number - 1].chunk
                pairs.append(
                    Pair(
                        question_id=question.id,
                        language=question.language,
                        sentence=sentence,
                        citation=number,
                        chunk_id=chunk.id,
                        chunk_text=chunk.text,
                    )
                )
    return pairs


class Judge(Protocol):
    name: str

    def verdict(self, pair: Pair) -> str: ...


def judge_prompt(pair: Pair) -> list[Message]:
    system = (
        "You check citations. You are given a source passage and one sentence from an answer that "
        "cites it. Reply with one word: supported if the passage states what the sentence claims, "
        "unsupported if the passage does not state it, unclear if you cannot tell. "
        "Judge against the passage only, never against what you already know. "
        "The sentence and the passage may be in different languages; that alone does not make it "
        "unsupported."
    )
    user = f"Passage:\n{pair.chunk_text}\n\nSentence:\n{pair.sentence}"
    return [Message("system", system), Message("user", user)]


def parse_verdict(text: str) -> str:
    lowered = text.strip().lower()
    # "unsupported" contains "supported", so it is tested first.
    for verdict in ("unsupported", "supported", "unclear"):
        if verdict in lowered:
            return verdict
    return "unclear"


class ModelJudge:
    def __init__(self, model: AnswerModel, name: str):
        self._model = model
        self.name = name

    def verdict(self, pair: Pair) -> str:
        return parse_verdict("".join(self._model.stream(judge_prompt(pair))))


def judge_all(pairs: Sequence[Pair], judge: Judge) -> list[dict]:
    return [
        {
            "question_id": pair.question_id,
            "language": pair.language,
            "citation": pair.citation,
            "chunk_id": pair.chunk_id,
            "sentence": pair.sentence,
            "chunk_text": pair.chunk_text,
            "verdict": judge.verdict(pair),
        }
        for pair in pairs
    ]


def faithfulness(judged: Sequence[dict]) -> dict:
    counts = Counter(record["verdict"] for record in judged)
    total = len(judged)
    return {
        "pairs": total,
        **{verdict: counts[verdict] for verdict in VERDICTS},
        "supported_rate": round(counts["supported"] / total, 4) if total else None,
    }
