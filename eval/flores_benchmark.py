"""FLORES-200 devtest benchmark, run once after model selection.

Usage:
    PYTHONUTF8=1 uv run --group train python -m eval.flores_benchmark

1. GlotLID (full and compressed) in its own label space over the whole devtest split, next to the
   published GlotLID and OpenLID figures, which also use devtest.
2. Every detector mapped onto the specialist's labels over the whole devtest split.
3. Contamination check: counts training-data rows equal to a FLORES-200 dev or devtest sentence or a
   Dakshina test line. Every count should be zero.
"""

import json
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from backend.pipeline.langid import FastTextScorer, LabelScorer
from backend.pipeline.normalize import normalize
from eval.detectors import UNDETERMINED, Detector, detector_specs
from training.data.clean import held_out_keys, text_key
from training.data.dakshina import dakshina_rows
from training.data.flores import CACHE_DIR, FLORES_SOURCE, fetch_flores, read_flores
from training.data.rows import read_jsonl
from training.env_report import environment
from training.labels import GROUPS, LABELS, OPENLID_LABELS, OTHER
from training.metrics import false_positive_rates, macro_f1, per_label_scores, report

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR = CACHE_DIR / "lid"
SPLITS = ("train", "val", "val_select", "test")

PUBLISHED = (
    {
        "model": "GlotLID-M",
        "source": "Kargaran et al. 2023 (arXiv:2310.16248), Table 2, FLORES-200 all",
        "split": "devtest",
        "threshold": 0.0,
        "languages": 196,
        "macro_f1": 0.917,
        "macro_fpr": 0.0042,
    },
    {
        "model": "GlotLID-M",
        "source": (
            "Kargaran et al. 2023 (arXiv:2310.16248), Table 3, FLORES-200, top half "
            '("SET?": language set not known to the model), '
            "averaged over the 195 languages shared with OpenLID"
        ),
        "split": "devtest",
        "threshold": 0.0,
        "languages": 195,
        "macro_f1": 0.916,
        "macro_fpr": 0.0043,
    },
    {
        "model": "OpenLID",
        "source": (
            "Kargaran et al. 2023 (arXiv:2310.16248), Table 3, FLORES-200, top half "
            '("SET?": language set not known to the model), '
            "averaged over the 195 languages shared with GlotLID-M"
        ),
        "split": "devtest",
        "threshold": 0.0,
        "languages": 195,
        "macro_f1": 0.923,
        "macro_fpr": 0.0051,
    },
    {
        "model": "OpenLID",
        "source": "Burchell et al. 2023 (arXiv:2305.13820), Table 1, FLORES-200*",
        "split": "devtest",
        "threshold": None,
        "languages": 201,
        "macro_f1": 0.927,
        "macro_fpr": 0.033,
        "note": "FPR as printed; the table does not say whether it is a fraction or a percentage",
    },
)
FLORES_TO_GLOTLID = {"zho_Hans": "cmn_Hani", "zho_Hant": "cmn_Hani", "yue_Hant": "yue_Hani"}
NOTES = (
    "GlotLID label space: zho_Hans and zho_Hant are both scored as cmn_Hani and yue_Hant as yue_Hani; "
    "FLORES codes that GlotLID lacks are excluded and listed per model.",
    "Specialist label space: the nine supported FLORES languages keep their codes and every other FLORES "
    "language counts as 'other'. GlotLID and lid.176 answers are mapped the same way, "
    "with the OpenCC rule for Han text.",
    "FLORES-200 has no romanized Hindi or Urdu; those labels are scored on the Dakshina test files "
    "in the langid-eval results.",
    "Published figures come from the papers' own model versions and label mappings; "
    "they are context, not a reproduction.",
)


def model_labels(scorer: LabelScorer) -> set[str]:
    return {candidate.code for candidate in scorer.top("a", k=-1)}


def scorer_codes(scorer: LabelScorer) -> Callable[[Sequence[str]], list[str]]:
    def predict(texts: Sequence[str]) -> list[str]:
        return [(top[0].code if (top := scorer.top(normalize(text), 1)) else UNDETERMINED) for text in texts]

    return predict


def native_space(
    predict: Callable[[Sequence[str]], list[str]], devtest: dict[str, list[str]], labels: set[str]
) -> dict:
    gold, texts, excluded = [], [], []
    for code, sentences in devtest.items():
        label = FLORES_TO_GLOTLID.get(code, code)
        if label not in labels:
            excluded.append(code)
            continue
        gold += [label] * len(sentences)
        texts += sentences
    predictions = predict(texts)
    evaluated = sorted(set(gold))
    scores = per_label_scores(gold, predictions, evaluated)
    rates = false_positive_rates(gold, predictions, evaluated)
    return {
        "languages": len(evaluated),
        "sentences": len(gold),
        "threshold": 0.0,
        "excluded_flores_codes": sorted(excluded),
        "merged_codes": FLORES_TO_GLOTLID,
        "macro_f1": round(macro_f1(scores, evaluated), 4),
        "macro_fpr": round(sum(rates.values()) / len(rates), 5) if rates else None,
        "per_label_f1": {label: round(scores[label].f1, 4) for label in evaluated},
    }


def specialist_space(detector: Detector, devtest: dict[str, list[str]]) -> dict:
    texts = [s for sentences in devtest.values() for s in sentences]
    gold = [
        code if code in OPENLID_LABELS else OTHER for code, sentences in devtest.items() for _ in sentences
    ]
    started = time.perf_counter()
    predictions = [label for label, _ in detector.predict_many(texts)]
    seconds = time.perf_counter() - started
    result = report(gold, predictions, LABELS, GROUPS)
    present = [label for label in LABELS if label in set(gold)]
    result["false_positive_rate"] = {
        k: round(v, 5) for k, v in false_positive_rates(gold, predictions, present).items()
    }
    result["seconds"] = round(seconds, 1)
    result["sentences_per_second"] = round(len(texts) / seconds, 1) if seconds else None
    return result


def contamination_check(data_dir: Path, held_out: dict[str, Sequence[str]]) -> dict[str, dict[str, int]]:
    keys = {name: held_out_keys(texts) for name, texts in held_out.items()}
    result: dict[str, dict[str, int]] = {name: {} for name in keys}
    for split in SPLITS:
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        row_keys = [text_key(normalize(row.text)) for row in read_jsonl(path)]
        for name, held in keys.items():
            result[name][split] = sum(key in held for key in row_keys)
    return result


def main() -> None:
    archive = fetch_flores()
    devtest, dev = read_flores(archive, "devtest"), read_flores(archive, "dev")
    manifest_path = MODELS_DIR / "MANIFEST.json"
    results = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": environment(),
        "benchmark": {
            **FLORES_SOURCE,
            "split": "devtest",
            "files": len(devtest),
            "sentences": sum(len(s) for s in devtest.values()),
        },
        "model_manifest": json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None,
        "published": list(PUBLISHED),
        "notes": list(NOTES),
        "native_space": {},
        "specialist_space": {},
    }
    for name, path in (
        ("glotlid-full", MODELS_DIR / "glotlid" / "model.bin"),
        ("glotlid-compressed", MODELS_DIR / "glotlid-q.ftz"),
    ):
        if path.exists():
            print(f"{name}: GlotLID label space")
            scorer = FastTextScorer(path)
            results["native_space"][name] = native_space(scorer_codes(scorer), devtest, model_labels(scorer))
            del scorer
    for spec in detector_specs(MODELS_DIR):
        print(f"{spec.name}: specialist label space")
        detector = spec.build()
        results["specialist_space"][spec.name] = specialist_space(detector, devtest)
        del detector
    held_out = {
        "flores200-dev": [s for sentences in dev.values() for s in sentences],
        "flores200-devtest": [s for sentences in devtest.values() for s in sentences],
    }
    if (CACHE_DIR / "dakshina" / "manifest.json").exists():
        _, sentences, words = dakshina_rows(CACHE_DIR / "dakshina")
        held_out["dakshina-test"] = [
            line for group in (sentences, words) for lines in group.values() for line in lines
        ]
    results["contamination"] = contamination_check(DATA_DIR, held_out)
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"lid-flores200-devtest-{datetime.now(UTC):%Y-%m-%d}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, entry in results["native_space"].items():
        print(
            f"{name}: macro-F1 {entry['macro_f1']} macro-FPR {entry['macro_fpr']} "
            f"over {entry['languages']} languages"
        )
    print(f"Contamination: {results['contamination']}")
    print(f"Wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
