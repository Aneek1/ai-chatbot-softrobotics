"""Train/validation/test splits separated by OpenLID's dataset_source (spec section 16.3).

For each label, the smallest sources are moved to test, then to validation, until each holds about
10% of the label's lines, as long as training keeps at least half. When no whole source fits (a label
with one dominant source, or the Dakshina labels), that split is carved from training by a seeded hash
of the text instead, and the split plan records that it is not source-disjoint.
"""

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from training.data.clean import text_key
from training.data.rows import Row

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class LabelSplit:
    label: str
    sources: dict[str, str]  # dataset_source -> "train" | "val" | "test"
    val_source_disjoint: bool
    test_source_disjoint: bool

    def as_dict(self) -> dict:
        return {
            "sources": dict(sorted(self.sources.items())),
            "val_source_disjoint": self.val_source_disjoint,
            "test_source_disjoint": self.test_source_disjoint,
        }


def hash_fraction(text: str, seed: int) -> float:
    digest = hashlib.blake2b(f"{seed}\x00{text}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def plan_label(
    label: str, source_counts: Mapping[str, int], target: float = 0.10, min_train: float = 0.50
) -> LabelSplit:
    total = sum(source_counts.values())
    assigned: dict[str, str] = {}
    filled = Counter()
    remaining = total
    for split in ("test", "val"):
        for source, count in sorted(source_counts.items(), key=lambda item: (item[1], item[0])):
            if source in assigned or filled[split] >= target * total:
                continue
            # A source bigger than twice the target would turn a 10% split into a much larger one.
            if count > 2 * target * total or remaining - count < min_train * total:
                continue
            assigned[source] = split
            filled[split] += count
            remaining -= count
    sources = {source: assigned.get(source, "train") for source in source_counts}
    return LabelSplit(
        label, sources, val_source_disjoint=filled["val"] > 0, test_source_disjoint=filled["test"] > 0
    )


def assign_splits(
    rows: Iterable[Row], seed: int, hash_only_labels: frozenset[str] = frozenset(), target: float = 0.10
) -> tuple[dict[str, list[Row]], dict[str, LabelSplit]]:
    rows = list(rows)
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        counts[row.label][row.dataset_source] += 1
    plans = {}
    for label, source_counts in sorted(counts.items()):
        if label in hash_only_labels:
            plans[label] = LabelSplit(label, dict.fromkeys(source_counts, "train"), False, False)
        else:
            plans[label] = plan_label(label, source_counts, target)

    splits: dict[str, list[Row]] = {name: [] for name in SPLITS}
    for row in rows:
        plan = plans[row.label]
        split = plan.sources[row.dataset_source]
        if split == "train":
            fraction = hash_fraction(row.text, seed)
            if not plan.test_source_disjoint and fraction < target:
                split = "test"
            elif not plan.val_source_disjoint and target <= fraction < 2 * target:
                split = "val"
        splits[split].append(row)
    return splits, plans


def remove_cross_split_duplicates(
    splits: Mapping[str, list[Row]],
) -> tuple[dict[str, list[Row]], dict[str, int]]:
    """Keep each text in the first split that has it (train, then val, then test), and once per split."""
    seen: set[bytes] = set()
    cleaned: dict[str, list[Row]] = {}
    removed: dict[str, int] = {}
    for name in SPLITS:
        kept = []
        for row in splits.get(name, []):
            key = text_key(row.text)
            if key in seen:
                continue
            seen.add(key)
            kept.append(row)
        removed[name] = len(splits.get(name, [])) - len(kept)
        cleaned[name] = kept
    return cleaned, removed
