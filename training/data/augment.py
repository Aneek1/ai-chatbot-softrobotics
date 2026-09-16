"""Synthetic rows (spec section 16.3), always tagged in Row.synthetic so results can be split."""

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from functools import lru_cache

import opencc

from training.data.rows import Row
from training.labels import OTHER

MAX_CROP_WORDS = 3
CROP_CHARS = (2, 8)


@lru_cache(maxsize=1)
def _converters() -> dict[str, tuple[str, opencc.OpenCC]]:
    return {"zho_Hans": ("zho_Hant", opencc.OpenCC("s2t")), "zho_Hant": ("zho_Hans", opencc.OpenCC("t2s"))}


def opencc_pairs(rows: Sequence[Row], rng: random.Random, per_label: int) -> list[Row]:
    """Convert a sample of Simplified lines to Traditional and the reverse; unchanged lines are skipped."""
    by_label = defaultdict(list)
    for row in rows:
        if row.label in _converters() and row.synthetic is None:
            by_label[row.label].append(row)
    pairs = []
    for label in sorted(by_label):
        target, converter = _converters()[label]
        pool = by_label[label]
        for row in rng.sample(pool, min(per_label, len(pool))):
            converted = converter.convert(row.text)
            if converted != row.text:
                pairs.append(replace(row, text=converted, label=target, synthetic="opencc"))
    return pairs


def crop(text: str, rng: random.Random) -> str:
    words = text.split()
    if len(words) >= 2:
        size = rng.randint(1, min(MAX_CROP_WORDS, len(words)))
        start = rng.randint(0, len(words) - size)
        return " ".join(words[start : start + size])
    # Chinese and Japanese lines usually have no spaces, so crop characters instead of words.
    low, high = CROP_CHARS
    if len(text) <= low:
        return text
    size = rng.randint(low, min(high, len(text)))
    start = rng.randint(0, len(text) - size)
    return text[start : start + size]


def short_crops(rows: Sequence[Row], rng: random.Random, fraction: float) -> list[Row]:
    natural = [row for row in rows if row.synthetic is None]
    chosen = rng.sample(natural, round(fraction * len(natural)))
    return [replace(row, text=crop(row.text, rng), synthetic="crop") for row in chosen]


def code_mixed(rows: Sequence[Row], rng: random.Random, count: int) -> list[Row]:
    """English joined with a sentence in another supported language. The label is the language
    that contributes more characters, since a single-label model can give only one answer."""
    natural = [row for row in rows if row.synthetic is None and row.label != OTHER]
    english = [row for row in natural if row.label == "eng_Latn"]
    others = [row for row in natural if row.label != "eng_Latn"]
    if not english or not others:
        return []
    mixed = []
    for _ in range(count):
        a, b = rng.choice(english), rng.choice(others)
        first, second = (a, b) if rng.random() < 0.5 else (b, a)
        label = a.label if len(a.text) > len(b.text) else b.label
        mixed.append(
            Row(
                text=f"{first.text} {second.text}",
                label=label,
                source=f"{first.source}+{second.source}",
                dataset_source=f"{first.dataset_source}+{second.dataset_source}",
                synthetic="code_mixed",
            )
        )
    return mixed
