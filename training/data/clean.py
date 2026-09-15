"""Cleaning (spec section 16.3): the app's normalization, lines under 2 characters dropped,
exact deduplication, and removal of any line that also appears in an evaluation set."""

import hashlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field, replace

from backend.pipeline.normalize import normalize
from training.data.rows import Row

MIN_CHARS = 2


def text_key(normalized_text: str) -> bytes:
    return hashlib.blake2b(normalized_text.encode("utf-8"), digest_size=16).digest()


def held_out_keys(texts: Iterable[str]) -> set[bytes]:
    return {text_key(normalize(text)) for text in texts}


@dataclass
class CleanStats:
    kept: Counter = field(default_factory=Counter)
    too_short: Counter = field(default_factory=Counter)
    duplicates: Counter = field(default_factory=Counter)
    label_conflicts: Counter = field(default_factory=Counter)
    contaminated: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict[str, dict[str, int]]:
        return {
            name: dict(sorted(getattr(self, name).items()))
            for name in ("kept", "too_short", "duplicates", "label_conflicts", "contaminated")
        }


class Cleaner:
    """Stateful so that every source shares one deduplication set and one contamination check."""

    def __init__(self, held_out: set[bytes] | None = None):
        self._held_out = held_out or set()
        self._seen: dict[bytes, str] = {}
        self.stats = CleanStats()

    def accept(self, row: Row) -> Row | None:
        text = normalize(row.text)
        if len(text) < MIN_CHARS:
            self.stats.too_short[row.label] += 1
            return None
        key = text_key(text)
        if key in self._held_out:
            self.stats.contaminated[row.label] += 1
            return None
        if key in self._seen:
            self.stats.duplicates[row.label] += 1
            if self._seen[key] != row.label:
                self.stats.label_conflicts[row.label] += 1
            return None
        self._seen[key] = row.label
        self.stats.kept[row.label] += 1
        return row if text == row.text else replace(row, text=text)


def clean_rows(rows: Iterable[Row], held_out: set[bytes] | None = None) -> tuple[list[Row], CleanStats]:
    cleaner = Cleaner(held_out)
    kept = [clean for row in rows if (clean := cleaner.accept(row)) is not None]
    return kept, cleaner.stats
