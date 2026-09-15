from collections import Counter

from training.data.rows import Row
from training.data.split import assign_splits, hash_fraction, plan_label, remove_cross_split_duplicates


def rows_for(label: str, source: str, count: int) -> list[Row]:
    return [Row(f"{label} {source} line {i}", label, "openlid", source) for i in range(count)]


def test_smallest_sources_go_to_test_then_val():
    plan = plan_label("eng_Latn", {"mt560": 80, "lti": 12, "tatoeba": 4, "wili2018": 11})
    assert plan.sources == {"mt560": "train", "lti": "val", "tatoeba": "test", "wili2018": "test"}
    assert plan.test_source_disjoint and plan.val_source_disjoint


def test_training_keeps_at_least_half_the_lines():
    plan = plan_label("zsm_Latn", {"leipzig": 60, "lti": 40})
    assert plan.sources == {"leipzig": "train", "lti": "train"}
    assert not plan.test_source_disjoint and not plan.val_source_disjoint


def test_single_source_label_is_hash_split():
    rows = rows_for("zho_Hant", "lti", 2000)
    splits, plans = assign_splits(rows, seed=7)
    assert plans["zho_Hant"].test_source_disjoint is False
    sizes = {name: len(split) for name, split in splits.items()}
    assert sum(sizes.values()) == 2000
    assert 150 < sizes["test"] < 250 and 150 < sizes["val"] < 250


def test_source_disjoint_label_keeps_sources_apart():
    rows = (
        rows_for("eng_Latn", "mt560", 80)
        + rows_for("eng_Latn", "lti", 12)
        + rows_for("eng_Latn", "tatoeba", 11)
    )
    splits, _ = assign_splits(rows, seed=7)
    sources = {name: Counter(r.dataset_source for r in split) for name, split in splits.items()}
    assert sources["train"] == {"mt560": 80}
    assert sources["val"] == {"lti": 12}
    assert sources["test"] == {"tatoeba": 11}


def test_hash_only_labels_ignore_sources():
    rows = rows_for("hin_Latn", "dev", 500) + rows_for("hin_Latn", "lexicon", 60)
    _, plans = assign_splits(rows, seed=7, hash_only_labels=frozenset({"hin_Latn"}))
    assert set(plans["hin_Latn"].sources.values()) == {"train"}
    assert plans["hin_Latn"].as_dict()["test_source_disjoint"] is False


def test_same_seed_same_split():
    rows = rows_for("kor_Hang", "lti", 300)
    first, _ = assign_splits(rows, seed=1)
    second, _ = assign_splits(rows, seed=1)
    assert first == second
    assert 0.0 <= hash_fraction("x", 1) < 1.0


def test_duplicates_across_splits_stay_in_the_earliest_split():
    shared = Row("same text", "eng_Latn", "openlid", "a")
    splits = {
        "train": [shared],
        "val": [shared, Row("val only", "eng_Latn", "openlid", "b")],
        "test": [shared],
    }
    cleaned, removed = remove_cross_split_duplicates(splits)
    assert cleaned["train"] == [shared]
    assert [r.text for r in cleaned["val"]] == ["val only"]
    assert cleaned["test"] == []
    assert removed == {"train": 0, "val": 1, "test": 1}
