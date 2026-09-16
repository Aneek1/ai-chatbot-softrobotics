"""Language-ID evaluation by slice (spec section 11.1) on held-out data.

Usage (after the models are exported):
    PYTHONUTF8=1 uv run --group train python -m eval.langid_eval
    PYTHONUTF8=1 uv run --group train python -m eval.langid_eval --detectors glotlid-compressed,lid.176

Sets: FLORES-200 devtest (Meta's original release; FLORES+ on Hugging Face is gated), the Dakshina
test files for romanized Hindi and Urdu, a sample of the internal test split, and the built
knowledge base, whose labels are the wiki or feed each document came from.
"""

import argparse
import json
import math
import random
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from eval.detectors import LID176_SOURCE, Detector, detector_specs
from eval.taiwan_vocab import CONFIG as TAIWAN_CONFIG
from eval.taiwan_vocab import taiwan_pairs
from training.data.augment import code_mixed, crop
from training.data.build import select_subset
from training.data.dakshina import DAKSHINA_SOURCE, dakshina_rows
from training.data.flores import CACHE_DIR, FLORES_SOURCE, fetch_flores, read_flores
from training.data.rows import Row, read_jsonl
from training.env_report import environment
from training.labels import GROUPS, LABELS, OPENLID_LABELS, OTHER
from training.metrics import report

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR = CACHE_DIR / "lid"
KNOWLEDGE_BASE = REPO_ROOT / "data" / "knowledge-base.jsonl"
FLORES_SET, DAKSHINA_SET, INTERNAL_SET = "flores200-devtest", "dakshina-test", "internal-test"
KB_SET = "knowledge-base"
TAIWAN_SLICE = "taiwan-vocabulary"
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
SEED = 20260915
OTHER_PER_LANGUAGE = 100
INTERNAL_PER_LABEL = 2_000
CODE_MIXED_COUNT = 1_000
TAIWAN_COUNT = 1_000
LATENCY_SAMPLES = 300


@dataclass(frozen=True)
class Item:
    text: str
    gold: str
    set: str
    slice: str


def flores_items(
    devtest: dict[str, list[str]], rng: random.Random, other_per_language: int = OTHER_PER_LANGUAGE
) -> list[Item]:
    items = []
    for code, sentences in devtest.items():
        if code in OPENLID_LABELS:
            items += [Item(s, code, FLORES_SET, "sentence") for s in sentences]
        else:
            items += [
                Item(s, OTHER, FLORES_SET, "sentence")
                for s in rng.sample(sentences, min(other_per_language, len(sentences)))
            ]
    return items


def short_items(items: Sequence[Item], rng: random.Random) -> list[Item]:
    return [Item(crop(i.text, rng), i.gold, i.set, "1-3 words") for i in items if i.gold != OTHER]


def paragraph_items(items: Sequence[Item]) -> list[Item]:
    by_label = defaultdict(list)
    for item in items:
        if item.gold != OTHER:
            by_label[item.gold].append(item)
    paragraphs = []
    for label, group in by_label.items():
        for start in range(0, len(group) - 2, 3):
            paragraphs.append(
                Item(" ".join(i.text for i in group[start : start + 3]), label, group[start].set, "paragraph")
            )
    return paragraphs


def code_mixed_items(items: Sequence[Item], rng: random.Random, count: int) -> list[Item]:
    rows = [Row(i.text, i.gold, "flores", i.set) for i in items if i.gold != OTHER]
    return [Item(row.text, row.label, FLORES_SET, "code-mixed") for row in code_mixed(rows, rng, count)]


def taiwan_items(
    devtest: dict[str, list[str]], rng: random.Random, count: int = TAIWAN_COUNT
) -> list[Item]:
    """Simplified FLORES lines and their Taiwan-vocabulary conversions, as one slice.

    Both members of a pair say the same thing, so the slice measures the Simplified/Traditional
    decision on its own. The Traditional side is an OpenCC conversion, not text written in Taiwan.
    """
    simplified = devtest.get("zho_Hans", [])
    sample = rng.sample(simplified, min(count, len(simplified)))
    items = []
    for source, converted in taiwan_pairs(sample):
        items.append(Item(source, "zho_Hans", FLORES_SET, TAIWAN_SLICE))
        items.append(Item(converted, "zho_Hant", FLORES_SET, TAIWAN_SLICE))
    return items


def knowledge_base_items(path: Path) -> list[Item]:
    """One item per indexed document, labelled with the language its source says it is written in.

    Nobody has checked those labels, and arXiv documents are English metadata about work in any
    language, so this set is reported next to the others, never merged into them.
    """
    if not path.exists():
        return []
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        document = json.loads(line)
        if document["language"] in LABELS:
            items.append(Item(document["text"], document["language"], KB_SET, document["source"]))
    return items


def dakshina_items(test_sentences: dict[str, list[str]], test_words: dict[str, list[str]]) -> list[Item]:
    items = [
        Item(s, label, DAKSHINA_SET, "sentence") for label, lines in test_sentences.items() for s in lines
    ]
    return items + [
        Item(w, label, DAKSHINA_SET, "1-3 words") for label, words in test_words.items() for w in words
    ]


def internal_items(rows: list[Row], rng: random.Random, per_label: int = INTERNAL_PER_LABEL) -> list[Item]:
    return [
        Item(r.text, r.label, INTERNAL_SET, f"synthetic:{r.synthetic}" if r.synthetic else "natural")
        for r in select_subset(rows, rng, per_label)
    ]


def threshold_sweep(
    gold: Sequence[str], predictions: Sequence[tuple[str, float]], thresholds: Sequence[float] = THRESHOLDS
) -> list[dict]:
    sweep = []
    for threshold in thresholds:
        covered = [
            (g, p) for g, (p, probability) in zip(gold, predictions, strict=True) if probability >= threshold
        ]
        sweep.append(
            {
                "threshold": threshold,
                "coverage": round(len(covered) / len(gold), 4) if gold else 0.0,
                "accuracy": round(sum(g == p for g, p in covered) / len(covered), 4) if covered else None,
            }
        )
    return sweep


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q / 100 * len(ordered)) - 1))]


def latency_ms(detector: Detector, texts: Sequence[str]) -> dict:
    timings = []
    for text in texts:
        started = time.perf_counter()
        detector.predict_many([text])
        timings.append((time.perf_counter() - started) * 1000)
    return {
        "samples": len(timings),
        "p50_ms": round(percentile(timings, 50), 3),
        "p95_ms": round(percentile(timings, 95), 3),
    }


def evaluate(detector: Detector, items: Sequence[Item]) -> dict:
    started = time.perf_counter()
    predictions = detector.predict_many([i.text for i in items])
    seconds = time.perf_counter() - started
    groups = defaultdict(list)
    for index, item in enumerate(items):
        groups[f"{item.set}/{item.slice}"].append(index)
        groups[f"{item.set}/all"].append(index)
    slices = {
        key: report([items[i].gold for i in index], [predictions[i][0] for i in index], LABELS, GROUPS)
        for key, index in sorted(groups.items())
    }
    sentences = [i for i, item in enumerate(items) if item.slice == "sentence"]
    return {
        "items": len(items),
        "seconds": round(seconds, 1),
        "slices": slices,
        "threshold_sweep": threshold_sweep(
            [items[i].gold for i in sentences], [predictions[i] for i in sentences]
        ),
    }


def compression_cost(full: dict, compressed: dict) -> dict[str, float]:
    return {
        label: round(compressed["per_label"][label]["f1"] - scores["f1"], 4)
        for label, scores in full["per_label"].items()
        if scores["support"]
    }


def rss_mib() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return round(psutil.Process().memory_info().rss / 2**20, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detectors", default="", help="comma-separated detector names; default: every detector found"
    )
    args = parser.parse_args()
    wanted = [name for name in args.detectors.split(",") if name]

    rng = random.Random(SEED)
    devtest = read_flores(fetch_flores(), "devtest")
    flores = flores_items(devtest, rng)
    taiwan = taiwan_items(devtest, rng)
    items = (
        flores
        + short_items(flores, rng)
        + paragraph_items(flores)
        + code_mixed_items(flores, rng, CODE_MIXED_COUNT)
        + taiwan
    )
    sets = {FLORES_SET: FLORES_SOURCE}
    dakshina_dir = CACHE_DIR / "dakshina"
    if (dakshina_dir / "manifest.json").exists():
        _, sentences, words = dakshina_rows(dakshina_dir)
        items += dakshina_items(sentences, words)
        manifest = json.loads((dakshina_dir / "manifest.json").read_text(encoding="utf-8"))
        sets[DAKSHINA_SET] = {**DAKSHINA_SOURCE, "archive_sha256": manifest["archive_sha256"]}
    if (DATA_DIR / "test.jsonl").exists():
        items += internal_items(list(read_jsonl(DATA_DIR / "test.jsonl")), rng)
        sets[INTERNAL_SET] = {"path": "data/cache/lid/test.jsonl", "per_label": INTERNAL_PER_LABEL}
    knowledge_base = knowledge_base_items(KNOWLEDGE_BASE)
    if knowledge_base:
        items += knowledge_base
        sets[KB_SET] = {
            "path": "data/knowledge-base.jsonl",
            "documents": len(knowledge_base),
            "labels": "the wiki or feed each document came from; not checked by a person",
        }
    latency_texts = [i.text for i in rng.sample(flores, LATENCY_SAMPLES)]

    results = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": environment(),
        "sets": sets,
        "items_per_set": {name: sum(i.set == name for i in items) for name in sets},
        "synthetic_slices": {
            TAIWAN_SLICE: {
                "opencc_config": TAIWAN_CONFIG,
                "built_from": "FLORES-200 devtest zho_Hans",
                "pairs": len(taiwan) // 2,
                "note": "conversions, not text written in Taiwan",
            }
        },
        "not_included": [
            "FLORES+ devtest (gated on Hugging Face); FLORES-200 devtest from Meta's release is used",
            "Curated fabrication methods: data/methods.csv holds its header row only, "
            "so the knowledge-base set is arXiv metadata and Wikipedia lead sections",
        ],
        "lid176": LID176_SOURCE,
        "detectors": {},
    }
    specs = [spec for spec in detector_specs(MODELS_DIR) if not wanted or spec.name in wanted]
    for spec in specs:
        before = rss_mib()
        detector = spec.build()
        after = rss_mib()
        print(f"Evaluating {spec.name} on {len(items):,} items")
        entry = evaluate(detector, items)
        entry["latency"] = latency_ms(detector, latency_texts)
        entry["model_bytes"] = sum(path.stat().st_size for path in spec.files if path.exists())
        entry["rss_increase_mib"] = (
            round(after - before, 1) if before is not None and after is not None else None
        )
        results["detectors"][spec.name] = entry
        del detector
    detectors = results["detectors"]
    if "glotlid-full" in detectors and "glotlid-compressed" in detectors:
        key = f"{FLORES_SET}/sentence"
        results["compression_cost_f1"] = compression_cost(
            detectors["glotlid-full"]["slices"][key], detectors["glotlid-compressed"]["slices"][key]
        )
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"langid-eval-{datetime.now(UTC):%Y-%m-%d}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
