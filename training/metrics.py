"""Classification metrics without third-party dependencies, so the fastText sweep
(which runs in an isolated environment) and the PyTorch runs report identically."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

OUTSIDE = "(outside group)"


@dataclass(frozen=True)
class LabelScores:
    precision: float
    recall: float
    f1: float
    support: int


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def per_label_scores(
    gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]
) -> dict[str, LabelScores]:
    if len(gold) != len(pred):
        raise ValueError(f"gold has {len(gold)} items but pred has {len(pred)}")
    true_pos, false_pos, false_neg = Counter(), Counter(), Counter()
    for g, p in zip(gold, pred, strict=True):
        if g == p:
            true_pos[g] += 1
        else:
            false_pos[p] += 1
            false_neg[g] += 1
    scores = {}
    for label in labels:
        precision = _ratio(true_pos[label], true_pos[label] + false_pos[label])
        recall = _ratio(true_pos[label], true_pos[label] + false_neg[label])
        f1 = _ratio(2 * precision * recall, precision + recall)
        scores[label] = LabelScores(precision, recall, f1, true_pos[label] + false_neg[label])
    return scores


def macro_f1(scores: Mapping[str, LabelScores], labels: Sequence[str] | None = None) -> float:
    """Mean F1 over labels that occur in the gold data; absent labels would add meaningless zeros."""
    chosen = [scores[label] for label in (labels if labels is not None else scores) if scores[label].support]
    return _ratio(sum(s.f1 for s in chosen), len(chosen))


def confusion_matrix(
    gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]
) -> dict[str, dict[str, int]]:
    """Rows are gold labels in `labels`; predictions outside `labels` are counted under OUTSIDE."""
    matrix = {g: {p: 0 for p in (*labels, OUTSIDE)} for g in labels}
    for g, p in zip(gold, pred, strict=True):
        if g in matrix:
            matrix[g][p if p in matrix[g] else OUTSIDE] += 1
    return matrix


def false_positive_rates(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> dict[str, float]:
    """FP / (FP + TN) per label, the second metric reported by the GlotLID and OpenLID papers."""
    total = len(gold)
    gold_counts = Counter(gold)
    false_pos = Counter(p for g, p in zip(gold, pred, strict=True) if g != p)
    return {label: _ratio(false_pos[label], total - gold_counts[label]) for label in labels}


def report(
    gold: Sequence[str],
    pred: Sequence[str],
    labels: Sequence[str],
    groups: Mapping[str, Sequence[str]],
) -> dict:
    scores = per_label_scores(gold, pred, labels)
    group_labels = [label for members in groups.values() for label in members]
    return {
        "count": len(gold),
        "accuracy": round(_ratio(sum(g == p for g, p in zip(gold, pred, strict=True)), len(gold)), 4),
        "macro_f1": round(macro_f1(scores, labels), 4),
        "group_macro_f1": round(macro_f1(scores, group_labels), 4),
        "group_f1": {name: round(macro_f1(scores, members), 4) for name, members in groups.items()},
        "per_label": {
            label: {
                "precision": round(s.precision, 4),
                "recall": round(s.recall, 4),
                "f1": round(s.f1, 4),
                "support": s.support,
            }
            for label, s in scores.items()
        },
        "confusions": {name: confusion_matrix(gold, pred, members) for name, members in groups.items()},
    }
