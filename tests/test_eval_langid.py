import random

import pytest

from eval.langid_eval import (
    Item,
    code_mixed_items,
    compression_cost,
    dakshina_items,
    evaluate,
    flores_items,
    internal_items,
    latency_ms,
    paragraph_items,
    percentile,
    short_items,
    threshold_sweep,
)
from training.data.rows import Row

DEVTEST = {
    "eng_Latn": [f"English sentence number {i} about soft robots" for i in range(7)],
    "fra_Latn": [f"Phrase numéro {i}" for i in range(10)],
    "zsm_Latn": ["Ayat bahasa Melayu yang agak panjang tentang robot lembut"],
}


class EchoDetector:
    """Predicts the gold label for everything except text containing 'Melayu'."""

    name = "echo"

    def __init__(self, gold: dict[str, str]):
        self._gold = gold
        self.calls = 0

    def predict_many(self, texts):
        self.calls += 1
        return [("ind_Latn", 0.4) if "Melayu" in t else (self._gold.get(t, "other"), 0.9) for t in texts]


def test_flores_items_keep_supported_codes_and_sample_the_rest():
    items = flores_items(DEVTEST, random.Random(0), other_per_language=3)
    golds = [i.gold for i in items]
    assert golds.count("eng_Latn") == 7 and golds.count("zsm_Latn") == 1 and golds.count("other") == 3
    assert {i.set for i in items} == {"flores200-devtest"} and {i.slice for i in items} == {"sentence"}


def test_short_crops_and_paragraphs_skip_other():
    items = flores_items(DEVTEST, random.Random(0), other_per_language=3)
    short = short_items(items, random.Random(1))
    assert len(short) == 8 and all(1 <= len(i.text.split()) <= 3 for i in short)
    paragraphs = paragraph_items(items)
    assert [p.gold for p in paragraphs] == ["eng_Latn", "eng_Latn"]
    assert paragraphs[0].text.count("English sentence") == 3


def test_code_mixed_items_are_tagged():
    items = flores_items(DEVTEST, random.Random(0), other_per_language=3)
    mixed = code_mixed_items(items, random.Random(2), count=5)
    assert len(mixed) == 5 and {i.slice for i in mixed} == {"code-mixed"}


def test_dakshina_items():
    items = dakshina_items({"hin_Latn": ["aap kaise hain"]}, {"urd_Latn": ["kitab"]})
    assert items == [
        Item("aap kaise hain", "hin_Latn", "dakshina-test", "sentence"),
        Item("kitab", "urd_Latn", "dakshina-test", "1-3 words"),
    ]


def test_internal_items_mark_synthetic_kinds():
    rows = [Row("soft robot", "eng_Latn", "openlid", "a"), Row("robot", "eng_Latn", "openlid", "a", "crop")]
    assert sorted(i.slice for i in internal_items(rows, random.Random(0), per_label=5)) == [
        "natural",
        "synthetic:crop",
    ]


def test_threshold_sweep():
    gold = ["a", "a", "b", "b"]
    predictions = [("a", 0.9), ("b", 0.4), ("b", 0.8), ("a", 0.2)]
    assert threshold_sweep(gold, predictions, (0.3, 0.5, 0.95)) == [
        {"threshold": 0.3, "coverage": 0.75, "accuracy": 0.6667},
        {"threshold": 0.5, "coverage": 0.5, "accuracy": 1.0},
        {"threshold": 0.95, "coverage": 0.0, "accuracy": None},
    ]


def test_percentile_nearest_rank():
    values = list(range(1, 101))
    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 95
    assert percentile([7.0], 95) == 7.0


def test_evaluate_reports_every_slice_and_the_threshold_sweep():
    items = flores_items(DEVTEST, random.Random(0), other_per_language=3)
    items += dakshina_items({"hin_Latn": ["aap kaise hain"]}, {})
    detector = EchoDetector({i.text: i.gold for i in items})
    result = evaluate(detector, items)
    assert set(result["slices"]) == {
        "dakshina-test/all",
        "dakshina-test/sentence",
        "flores200-devtest/all",
        "flores200-devtest/sentence",
    }
    flores = result["slices"]["flores200-devtest/sentence"]
    assert flores["per_label"]["zsm_Latn"]["recall"] == 0.0
    assert flores["confusions"]["malay_indonesian"]["zsm_Latn"]["ind_Latn"] == 1
    assert result["threshold_sweep"][2] == {
        "threshold": 0.5,
        "coverage": pytest.approx(11 / 12, abs=1e-4),
        "accuracy": 1.0,
    }
    assert detector.calls == 1


def test_compression_cost_skips_labels_without_support():
    full = {"per_label": {"eng_Latn": {"f1": 0.99, "support": 10}, "kor_Hang": {"f1": 0.0, "support": 0}}}
    compressed = {
        "per_label": {"eng_Latn": {"f1": 0.97, "support": 10}, "kor_Hang": {"f1": 0.0, "support": 0}}
    }
    assert compression_cost(full, compressed) == {"eng_Latn": -0.02}


def test_latency_times_single_texts():
    detector = EchoDetector({})
    result = latency_ms(detector, ["a", "b", "c"])
    assert result["samples"] == 3 and detector.calls == 3
    assert 0 <= result["p50_ms"] <= result["p95_ms"]
