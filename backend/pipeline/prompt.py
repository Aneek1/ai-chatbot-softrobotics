from collections.abc import Sequence

from backend.pipeline.languages import display_name
from backend.pipeline.retrieve import Hit
from backend.providers.base import Message


def build_messages(question: str, answer_language: str, hits: Sequence[Hit]) -> list[Message]:
    sources = "\n\n".join(f"[{i}] {h.chunk.title}\n{h.chunk.text}" for i, h in enumerate(hits, start=1))
    system = (
        "You answer questions about soft-robotics fabrication. "
        f"Answer in {display_name(answer_language)}.\n"
        "Use only the numbered sources below and cite them as [n] after the sentence they support.\n"
        "If the sources do not cover the question, say so plainly instead of guessing.\n\n"
        f"Sources:\n{sources or '(none)'}"
    )
    return [Message("system", system), Message("user", question)]
