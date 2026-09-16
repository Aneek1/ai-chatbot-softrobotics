import re
from dataclasses import dataclass

_CITATION = re.compile(r"\s?\[(\d+)\]")


@dataclass(frozen=True)
class CitationCheck:
    text: str
    valid: tuple[int, ...]
    removed: tuple[int, ...]


def check_citations(answer: str, source_count: int) -> CitationCheck:
    valid: list[int] = []
    removed: list[int] = []

    def replace(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if 1 <= number <= source_count:
            if number not in valid:
                valid.append(number)
            return match.group(0)
        if number not in removed:
            removed.append(number)
        return ""

    return CitationCheck(text=_CITATION.sub(replace, answer), valid=tuple(valid), removed=tuple(removed))
