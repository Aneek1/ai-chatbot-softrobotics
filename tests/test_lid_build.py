from collections import Counter

from training.data.build import BuildConfig, build_dataset, count_table, select_subset
from training.data.card import render_card
from training.data.clean import text_key
from training.data.rows import Row
from training.labels import LABELS

CONFIG = BuildConfig(
    seed=11, opencc_per_label=50, crop_fraction=0.1, code_mixed_fraction=0.05, select_per_label=15
)


def openlid_rows() -> list[Row]:
    rows = []
    samples = {
        "eng_Latn": "soft robot arm number {i} bends when air flows in",
        "ind_Latn": "lengan robot lunak nomor {i} menekuk saat udara masuk",
        "zsm_Latn": "lengan robot lembut nombor {i} membengkok apabila udara masuk",
        "zho_Hans": "软体机器人手臂第{i}号在空气流入时弯曲",
        "zho_Hant": "矽膠夾爪第{i}號由兩層彈性體製成",
        "other": "le bras robotique souple numéro {i} se plie",
    }
    for label, template in samples.items():
        for i in range(120):
            source = ("mt560", "lti", "tatoeba")[0 if i < 90 else (1 if i < 106 else 2)]
            rows.append(Row(template.format(i=i), label, "openlid", source))
    return rows


def dakshina_rows() -> list[Row]:
    return [Row(f"aap kaise hain {i}", "hin_Latn", "dakshina", "hi.dev") for i in range(60)] + [
        Row(f"aap kaisay hain {i}", "urd_Latn", "dakshina", "ur.dev") for i in range(60)
    ]


def test_splits_are_disjoint_and_tagged():
    dataset = build_dataset(openlid_rows(), dakshina_rows(), CONFIG)
    keys = {name: {text_key(r.text) for r in dataset.splits[name]} for name in ("train", "val", "test")}
    assert not keys["train"] & keys["val"]
    assert not keys["train"] & keys["test"]
    assert not keys["val"] & keys["test"]
    kinds = Counter(r.synthetic for r in dataset.splits["train"])
    assert kinds[None] > 0 and kinds["opencc"] > 0 and kinds["crop"] > 0 and kinds["code_mixed"] > 0


def test_openlid_sources_stay_out_of_training_when_they_can():
    dataset = build_dataset(openlid_rows(), dakshina_rows(), CONFIG)
    natural_test_sources = {
        r.dataset_source for r in dataset.splits["test"] if r.synthetic is None and r.source == "openlid"
    }
    assert natural_test_sources == {"tatoeba"}
    assert dataset.report["split_plans"]["eng_Latn"]["test_source_disjoint"] is True
    assert dataset.report["split_plans"]["hin_Latn"]["test_source_disjoint"] is False


def test_val_select_is_capped_per_label_and_drawn_from_val():
    dataset = build_dataset(openlid_rows(), dakshina_rows(), CONFIG)
    per_label = Counter(r.label for r in dataset.splits["val_select"])
    assert max(per_label.values()) <= 15
    assert set(dataset.splits["val_select"]) <= set(dataset.splits["val"])


def test_build_is_deterministic():
    first = build_dataset(openlid_rows(), dakshina_rows(), CONFIG)
    second = build_dataset(openlid_rows(), dakshina_rows(), CONFIG)
    assert first.splits == second.splits


def test_count_table_separates_natural_and_synthetic():
    rows = {"train": [Row("a b", "eng_Latn", "openlid", "x"), Row("a", "eng_Latn", "openlid", "x", "crop")]}
    assert count_table(rows) == {"train": {"eng_Latn": {"crop": 1, "natural": 1}}}


def test_select_subset_keeps_order():
    import random

    rows = [Row(f"t{i}", "eng_Latn", "openlid", "x") for i in range(10)]
    chosen = select_subset(rows, random.Random(0), per_label=4)
    assert len(chosen) == 4
    assert chosen == sorted(chosen, key=rows.index)


def test_card_reports_sources_counts_and_limits():
    dataset = build_dataset(openlid_rows(), dakshina_rows(), CONFIG)
    report = {
        "built_at": "2026-09-15T22:00:00+00:00",
        "sources": {
            "openlid": {
                "url": "https://huggingface.co/datasets/laurievb/open-lid-dataset",
                "revision": "67b9a87a9f52285a666aea50a0e118b8db8f6daa",
                "shards_used": ["a", "b"],
                "shards_available": 44,
                "licence": "other",
            },
            "dakshina": {
                "url": "https://example.invalid/dakshina",
                "archive_sha256": "ab" * 32,
                "licence": "CC-BY-SA-4.0",
            },
            "flores": {
                "url": "https://example.invalid/flores",
                "sha256": "cd" * 32,
                "licence": "CC-BY-SA-4.0",
            },
        },
        "openlid_below_cap": {"zsm_Latn": 81234},
        "cleaning": {
            "too_short": {"eng_Latn": 3},
            "duplicates": {"eng_Latn": 5},
            "contaminated": {"eng_Latn": 2},
        },
        "held_out": {"flores_sentences": 412212, "dakshina_test_lines": 12000},
        **dataset.report,
    }
    card = render_card(report)
    assert "seed 11" in card
    assert "2 of 44 shards" in card
    assert "`zsm_Latn` (81,234)" in card
    assert "2 lines removed" in card
    assert "412,212 checked" in card
    for label in LABELS:
        assert f"| `{label}` |" in card
    assert "| `eng_Latn` | tatoeba | lti | yes | yes |" in card
    assert "OpenCC conversions do not read like natively written" in card


def test_synthetic_rows_matching_held_out_lines_are_removed():
    from training.data.clean import held_out_keys

    words = "soft robot arm bends when air flows in".split()
    held_out = held_out_keys(words + [f"{a} {b}" for a in words for b in words])
    dataset = build_dataset(openlid_rows(), dakshina_rows(), CONFIG, held_out)
    synthetic = {r.text for split in ("train", "val", "test") for r in dataset.splits[split] if r.synthetic}
    assert not synthetic & set(words)
    assert sum(dataset.report["synthetic_contaminated_removed"].values()) > 0
