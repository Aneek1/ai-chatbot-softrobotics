"""Generate training/RESULTS.md from the committed results files, so every number in it has a source.

Usage:
    uv run python -m training.results_table
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT = REPO_ROOT / "training" / "RESULTS.md"
GROUP_NAMES = ("malay_indonesian", "han", "romanized_hindi")
SLICES = (
    ("flores200-devtest/sentence", "FLORES sentence"),
    ("flores200-devtest/1-3 words", "FLORES 1-3 words"),
    ("flores200-devtest/paragraph", "FLORES paragraph"),
    ("flores200-devtest/taiwan-vocabulary", "Taiwan vocabulary"),
    ("flores200-devtest/code-mixed", "code-mixed"),
    ("dakshina-test/sentence", "Dakshina sentence"),
    ("dakshina-test/1-3 words", "Dakshina words"),
)


def fmt(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def latest(pattern: str, results_dir: Path = RESULTS_DIR) -> Path | None:
    matches = sorted(results_dir.glob(pattern))
    return matches[-1] if matches else None


def validation_section(selection: dict, source: str) -> list[str]:
    lines = [
        "## Validation (model selection)",
        "",
        f"Source: `{source}`. Metric: {selection['selection_metric']}. Chosen: `{selection['overall']}`.",
        "",
        "| Run | Family | Group macro-F1 | Macro-F1 | " + " | ".join(GROUP_NAMES) + " | Model MB | Minutes |",
        "|---|---|---:|---:|" + "---:|" * len(GROUP_NAMES) + "---:|---:|",
    ]
    for run in selection["ranking"]:
        groups = " | ".join(fmt(run["val_group_f1"].get(name)) for name in GROUP_NAMES)
        lines.append(
            f"| `{run['run_id']}` | {run['family']} | {fmt(run['val_group_macro_f1'])} | "
            f"{fmt(run['val_macro_f1'])} | "
            f"{groups} | {run['model_bytes'] / 1e6:.1f} | {run['wall_clock_seconds'] / 60:.0f} |"
        )
    return lines + [""]


def comparison_sentence(benchmark: dict) -> str | None:
    ours = benchmark["native_space"].get("glotlid-full")
    if ours is None:
        return None
    published = benchmark["published"][0]
    difference = ours["macro_f1"] - published["macro_f1"]
    relation = (
        "within 0.01 of"
        if abs(difference) <= 0.01
        else f"{abs(difference):.3f} {'above' if difference > 0 else 'below'}"
    )
    return (
        f"Our glotlid-full run gives macro-F1 {ours['macro_f1']:.3f} over {ours['languages']} languages, "
        f"{relation} "
        f"the {published['macro_f1']:.3f} over {published['languages']} languages in {published['source']}. "
        "The model revision in models/ and the label mapping may differ from the paper's, "
        "so this is a consistency "
        "check, not a reproduction."
    )


def native_section(benchmark: dict, source: str) -> list[str]:
    lines = [
        "## FLORES-200 devtest, GlotLID label space",
        "",
        f"Source: `{source}` ({benchmark['benchmark']['sentences']:,} sentences, threshold 0).",
        "",
        "| Model | Figures from | Languages | Macro-F1 | Macro-FPR |",
        "|---|---|---:|---:|---:|",
    ]
    for name, entry in benchmark["native_space"].items():
        lines.append(
            f"| {name} | this repository | {entry['languages']} | {fmt(entry['macro_f1'])} | "
            f"{fmt(entry['macro_fpr'], 4)} |"
        )
    for entry in benchmark["published"]:
        note = f" ({entry['note']})" if entry.get("note") else ""
        lines.append(
            f"| {entry['model']} | {entry['source']}{note} | {entry['languages']} | "
            f"{fmt(entry['macro_f1'])} | "
            f"{entry['macro_fpr']} |"
        )
    sentence = comparison_sentence(benchmark)
    return lines + ([""] + [sentence] if sentence else []) + [""]


def specialist_section(benchmark: dict, source: str) -> list[str]:
    lines = [
        "## FLORES-200 devtest, specialist labels",
        "",
        f"Source: `{source}`. Every language outside the nine supported FLORES codes counts as `other`. "
        "FLORES-200 has no romanized Hindi or Urdu, "
        "so the romanized_hindi column here reflects English only.",
        "",
        "| Detector | Macro-F1 | Group macro-F1 | " + " | ".join(GROUP_NAMES) + " | other F1 | Sentences/s |",
        "|---|---:|---:|" + "---:|" * len(GROUP_NAMES) + "---:|---:|",
    ]
    for name, entry in benchmark["specialist_space"].items():
        groups = " | ".join(fmt(entry["group_f1"].get(group)) for group in GROUP_NAMES)
        lines.append(
            f"| {name} | {fmt(entry['macro_f1'])} | {fmt(entry['group_macro_f1'])} | {groups} | "
            f"{fmt(entry['per_label']['other']['f1'])} | {fmt(entry['sentences_per_second'], 0)} |"
        )
    return lines + [""]


def slices_section(evaluation: dict, source: str) -> list[str]:
    lines = [
        "## Macro-F1 by slice",
        "",
        f"Source: `{source}`.",
        "",
        "| Detector | " + " | ".join(title for _, title in SLICES) + " | p50 ms | p95 ms | Model MB |",
        "|---|" + "---:|" * len(SLICES) + "---:|---:|---:|",
    ]
    for name, entry in evaluation["detectors"].items():
        cells = " | ".join(fmt(entry["slices"].get(key, {}).get("macro_f1")) for key, _ in SLICES)
        lines.append(
            f"| {name} | {cells} | {fmt(entry['latency']['p50_ms'], 2)} | "
            f"{fmt(entry['latency']['p95_ms'], 2)} | "
            f"{entry['model_bytes'] / 1e6:.1f} |"
        )
    return lines + [""]


def contamination_section(benchmark: dict) -> list[str]:
    counts = benchmark["contamination"]
    total = sum(n for splits in counts.values() for n in splits.values())
    detail = "; ".join(
        f"{name}: " + ", ".join(f"{split} {n}" for split, n in splits.items())
        for name, splits in counts.items()
    )
    return ["## Contamination check", "", f"{total} dataset rows match a held-out line ({detail}).", ""]


def render(
    selection: tuple[dict, str] | None,
    evaluation: tuple[dict, str] | None,
    benchmark: tuple[dict, str] | None,
) -> str:
    lines = [
        "# Language-ID results",
        "",
        "Generated by `training/results_table.py` from the files in `results/`; do not edit by hand.",
        "",
    ]
    if selection:
        lines += validation_section(*selection)
    if benchmark:
        lines += native_section(*benchmark) + specialist_section(*benchmark)
    if evaluation:
        lines += slices_section(*evaluation)
    if benchmark:
        lines += (
            contamination_section(benchmark[0])
            + ["## Notes", ""]
            + [f"- {note}" for note in benchmark[0]["notes"]]
            + [""]
        )
    return "\n".join(lines)


def _load(pattern: str) -> tuple[dict, str] | None:
    path = latest(pattern)
    return (json.loads(path.read_text(encoding="utf-8")), f"results/{path.name}") if path else None


def main() -> None:
    OUTPUT.write_text(
        render(
            _load("lid-selection-*.json"), _load("langid-eval-*.json"), _load("lid-flores200-devtest-*.json")
        ),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
