"""Build the language-ID dataset (spec section 16.3) into data/cache/lid/.

Usage:
    PYTHONUTF8=1 uv run --group train python -m training.data.build

Downloads FLORES-200 (25.6 MB, contamination check), streams Dakshina (2.0 GB, keeps about 5 MB)
and reads OpenLID shards (about 376 MB each) one at a time, deleting each after sampling.
"""

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from backend.pipeline.normalize import normalize
from training.data.augment import code_mixed, opencc_pairs, short_crops
from training.data.card import render_card
from training.data.clean import Cleaner, held_out_keys, text_key
from training.data.dakshina import DAKSHINA_SOURCE, dakshina_rows, fetch_dakshina
from training.data.flores import CACHE_DIR, FLORES_SOURCE, fetch_flores, read_flores
from training.data.openlid import (
    OPENLID_SOURCE,
    OpenLIDSampler,
    download_shard,
    iter_parquet,
    list_shards,
    sample_shards,
    shard_order,
)
from training.data.rows import Row, write_jsonl
from training.data.split import SPLITS, assign_splits, remove_cross_split_duplicates
from training.labels import DAKSHINA_LABELS, OPENLID_LABELS

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = CACHE_DIR / "lid"
CARD_PATH = REPO_ROOT / "training" / "data" / "DATACARD.md"
RESULTS_DIR = REPO_ROOT / "results"
HAN_LABELS = ("zho_Hans", "zho_Hant")


@dataclass(frozen=True)
class BuildConfig:
    seed: int = 20260915
    target_cap: int = 100_000
    other_cap: int = 1_000
    max_shards: int = 16
    split_target: float = 0.10
    opencc_per_label: int = 20_000
    opencc_fraction: float = 0.20
    crop_fraction: float = 0.10
    code_mixed_fraction: float = 0.02
    select_per_label: int = 2_000


@dataclass
class Dataset:
    splits: dict[str, list[Row]]
    report: dict


def augment_split(rows: list[Row], rng: random.Random, config: BuildConfig) -> list[Row]:
    natural = [row for row in rows if row.synthetic is None]
    han = sum(row.label in HAN_LABELS for row in natural)
    per_label = min(config.opencc_per_label, round(config.opencc_fraction * han / 2))
    synthetic = opencc_pairs(natural, rng, per_label)
    synthetic += short_crops(natural, rng, config.crop_fraction)
    synthetic += code_mixed(natural, rng, round(config.code_mixed_fraction * len(natural)))
    return rows + synthetic


def select_subset(rows: list[Row], rng: random.Random, per_label: int) -> list[Row]:
    positions = defaultdict(list)
    for position, row in enumerate(rows):
        positions[row.label].append(position)
    chosen = sorted(
        p
        for label in sorted(positions)
        for p in rng.sample(positions[label], min(per_label, len(positions[label])))
    )
    return [rows[p] for p in chosen]


def count_table(splits: dict[str, list[Row]]) -> dict[str, dict[str, dict[str, int]]]:
    table: dict[str, dict[str, Counter]] = {}
    for name, rows in splits.items():
        per_label: dict[str, Counter] = defaultdict(Counter)
        for row in rows:
            per_label[row.label][row.synthetic or "natural"] += 1
        table[name] = {label: dict(sorted(c.items())) for label, c in sorted(per_label.items())}
    return table


def build_dataset(
    openlid_rows: list[Row],
    dakshina_train_rows: list[Row],
    config: BuildConfig,
    held_out: set[bytes] | None = None,
) -> Dataset:
    rng = random.Random(config.seed)
    splits, plans = assign_splits(
        openlid_rows + dakshina_train_rows,
        config.seed,
        hash_only_labels=frozenset(DAKSHINA_LABELS.values()),
        target=config.split_target,
    )
    splits, natural_overlap = remove_cross_split_duplicates(splits)
    augmented = {name: augment_split(splits[name], rng, config) for name in SPLITS}
    # Natural rows were checked while sampling; a synthetic crop can still equal a held-out line
    # (a one-word crop matching a Dakshina test word, for example).
    synthetic_contaminated = {}
    for name in SPLITS:
        kept = [
            r
            for r in augmented[name]
            if r.synthetic is None or text_key(normalize(r.text)) not in (held_out or ())
        ]
        synthetic_contaminated[name] = len(augmented[name]) - len(kept)
        augmented[name] = kept
    # Synthetic rows can recreate a line from another split (a one-word crop, an OpenCC conversion).
    augmented, synthetic_overlap = remove_cross_split_duplicates(augmented)
    augmented["val_select"] = select_subset(augmented["val"], rng, config.select_per_label)
    report = {
        "config": asdict(config),
        "split_plans": {label: plan.as_dict() for label, plan in sorted(plans.items())},
        "cross_split_duplicates_removed": {
            "natural": natural_overlap,
            "after_augmentation": synthetic_overlap,
        },
        "synthetic_contaminated_removed": synthetic_contaminated,
        "counts": count_table(augmented),
    }
    return Dataset(augmented, report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=BuildConfig.seed)
    parser.add_argument("--target-cap", type=int, default=BuildConfig.target_cap)
    parser.add_argument("--other-cap", type=int, default=BuildConfig.other_cap)
    parser.add_argument("--max-shards", type=int, default=BuildConfig.max_shards)
    parser.add_argument(
        "--keep-shards", action="store_true", help="keep downloaded OpenLID shards in data/cache"
    )
    args = parser.parse_args()
    config = BuildConfig(
        seed=args.seed, target_cap=args.target_cap, other_cap=args.other_cap, max_shards=args.max_shards
    )
    started = time.monotonic()

    flores_archive = fetch_flores()
    flores_sentences = [
        s
        for split in ("dev", "devtest")
        for lines in read_flores(flores_archive, split).values()
        for s in lines
    ]
    dakshina_dir = CACHE_DIR / "dakshina"
    dakshina_manifest = fetch_dakshina(dakshina_dir)
    dakshina_train, test_sentences, test_words = dakshina_rows(dakshina_dir)
    dakshina_test = [
        line for group in (test_sentences, test_words) for lines in group.values() for line in lines
    ]
    print(
        f"Held out: {len(flores_sentences):,} FLORES-200 sentences, "
        f"{len(dakshina_test):,} Dakshina test lines"
    )

    held_out = held_out_keys(flores_sentences + dakshina_test)
    cleaner = Cleaner(held_out)
    dakshina_clean = [row for original in dakshina_train if (row := cleaner.accept(original)) is not None]

    available = list_shards()
    ordered = shard_order(available, config.seed)[: config.max_shards]
    sampler = OpenLIDSampler(cleaner, config.target_cap, config.other_cap)

    def read_and_delete(path: str):
        local = download_shard(path, CACHE_DIR)
        try:
            yield from iter_parquet(local)
        finally:
            if not args.keep_shards:
                local.unlink(missing_ok=True)

    used = sample_shards((read_and_delete(path) for path in ordered), sampler)
    dataset = build_dataset(sampler.rows, dakshina_clean, config, held_out)
    for name, rows in dataset.splits.items():
        print(f"{name}: {write_jsonl(rows, args.out / f'{name}.jsonl'):,} rows")

    built_at = datetime.now(UTC)
    report = {
        "built_at": built_at.isoformat(timespec="seconds"),
        "sources": {
            "openlid": {**OPENLID_SOURCE, "shards_used": ordered[:used], "shards_available": len(available)},
            "dakshina": {
                **DAKSHINA_SOURCE,
                "archive_sha256": dakshina_manifest["archive_sha256"],
                "files": dakshina_manifest["files"],
            },
            "flores": FLORES_SOURCE,
        },
        "openlid_rows_per_language": dict(sorted(sampler.counts.items())),
        "openlid_below_cap": {
            label: sampler.counts[label]
            for label in sorted(OPENLID_LABELS)
            if sampler.counts[label] < config.target_cap
        },
        "cleaning": cleaner.stats.as_dict(),
        "held_out": {"flores_sentences": len(flores_sentences), "dakshina_test_lines": len(dakshina_test)},
        **dataset.report,
        "wall_clock_seconds": round(time.monotonic() - started, 1),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    RESULTS_DIR.mkdir(exist_ok=True)
    results_path = RESULTS_DIR / f"lid-data-{built_at:%Y-%m-%d}.json"
    results_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    CARD_PATH.write_text(render_card(report), encoding="utf-8")
    print(f"Wrote {results_path.relative_to(REPO_ROOT)} and {CARD_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
