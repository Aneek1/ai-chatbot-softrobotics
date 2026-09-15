from backend.pipeline.chunk import chunk_text, split_sentences
from tests.fakes import word_count


def test_latin_sentences_split_on_terminal_punctuation():
    assert split_sentences("One two. Three four! Five?", "Latn") == ["One two.", "Three four!", "Five?"]


def test_cjk_sentences_split_without_spaces():
    assert split_sentences("第一句。第二句！第三句？", "Hani") == ["第一句。", "第二句！", "第三句？"]


def test_packs_sentences_with_overlap():
    text = "One two three. Four five six. Seven eight nine."
    assert chunk_text(text, "Latn", word_count, max_tokens=6, overlap_tokens=3) == [
        "One two three. Four five six.",
        "Four five six. Seven eight nine.",
    ]


def test_cjk_chunks_join_without_spaces():
    chunks = chunk_text("第一句。第二句！第三句？", "Hani", len, max_tokens=8, overlap_tokens=0)
    assert chunks == ["第一句。第二句！", "第三句？"]


def test_long_sentence_is_split():
    assert chunk_text("a b c d e f g", "Latn", word_count, max_tokens=3, overlap_tokens=0) == [
        "a b c",
        "d e f",
        "g",
    ]


def test_overlap_dropped_when_it_would_overflow():
    text = "a b. c d e f."
    chunks = chunk_text(text, "Latn", word_count, max_tokens=4, overlap_tokens=2)
    assert chunks == ["a b.", "c d e f."]
    assert all(word_count(c) <= 4 for c in chunks)


def test_empty_text():
    assert chunk_text("", "Latn", word_count) == []
