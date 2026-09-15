import re
from collections.abc import Callable
from pathlib import Path

TokenCounter = Callable[[str], int]

CJK_SCRIPTS = frozenset({"Hani", "Jpan"})
_CJK_SENTENCE_END = re.compile(r"(?<=[。！？!?])")
_SENTENCE_END = re.compile(r"(?<=[.!?।॥])\s+")


def _joiner(script: str | None) -> str:
    return "" if script in CJK_SCRIPTS else " "


def split_sentences(text: str, script: str | None) -> list[str]:
    pattern = _CJK_SENTENCE_END if script in CJK_SCRIPTS else _SENTENCE_END
    return [s.strip() for s in pattern.split(text) if s.strip()]


def _split_long(sentence: str, script: str | None, count: TokenCounter, max_tokens: int) -> list[str]:
    units = list(sentence) if script in CJK_SCRIPTS else sentence.split(" ")
    joiner = _joiner(script)
    pieces: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and count(joiner.join([*current, unit])) > max_tokens:
            pieces.append(joiner.join(current))
            current = [unit]
        else:
            current.append(unit)
    if current:
        pieces.append(joiner.join(current))
    return pieces


def _overlap_tail(sentences: list[str], joiner: str, count: TokenCounter, overlap_tokens: int) -> list[str]:
    tail: list[str] = []
    for sentence in reversed(sentences):
        if count(joiner.join([sentence, *tail])) > overlap_tokens:
            break
        tail.insert(0, sentence)
    return tail


def chunk_text(
    text: str,
    script: str | None,
    count: TokenCounter,
    max_tokens: int = 300,
    overlap_tokens: int = 50,
) -> list[str]:
    joiner = _joiner(script)
    sentences: list[str] = []
    for sentence in split_sentences(text, script):
        if count(sentence) > max_tokens:
            sentences.extend(_split_long(sentence, script, count, max_tokens))
        else:
            sentences.append(sentence)

    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if current and count(joiner.join([*current, sentence])) > max_tokens:
            chunks.append(joiner.join(current))
            current = _overlap_tail(current, joiner, count, overlap_tokens)
            if current and count(joiner.join([*current, sentence])) > max_tokens:
                current = []
        current.append(sentence)
    if current:
        chunks.append(joiner.join(current))
    return chunks


def load_token_counter(tokenizer_path: Path) -> TokenCounter:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
