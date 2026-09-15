from backend.pipeline.prompt import build_messages
from backend.pipeline.retrieve import Chunk, Hit


def hit(n, text):
    return Hit(
        chunk=Chunk(
            f"d{n}:0", f"d{n}", text, "eng_Latn", "fixture",
            f"Title {n}", "https://example.invalid", "test-only",
        ),
        score=0.9,
    )


def test_system_prompt_names_language_and_numbers_sources():
    messages = build_messages("Bagaimana?", "ind_Latn", [hit(1, "first"), hit(2, "second")])
    system, user = messages
    assert system.role == "system" and user.role == "user"
    assert "Answer in Indonesian." in system.content
    assert "[1] Title 1\nfirst" in system.content
    assert "[2] Title 2\nsecond" in system.content
    assert user.content == "Bagaimana?"


def test_no_sources_is_stated():
    system, _ = build_messages("Hi", "eng_Latn", [])
    assert "Sources:\n(none)" in system.content
