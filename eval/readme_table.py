"""Generate the results block in README.md from the committed evaluation files.

Usage:
    PYTHONUTF8=1 uv run python -m eval.readme_table

Every number in the README's results section comes from this script. The fuller tables, including
the threshold sweep and the comparison with published figures, stay in training/RESULTS.md.
"""

import json
from pathlib import Path

from training.results_table import fmt, latest

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
START = "<!-- results:start -->"
END = "<!-- results:end -->"
SLICES = (
    ("flores200-devtest/sentence", "FLORES sentence"),
    ("flores200-devtest/1-3 words", "FLORES 1-3 words"),
    ("flores200-devtest/code-mixed", "code-mixed"),
    ("dakshina-test/sentence", "Dakshina sentence"),
    ("internal-test/all", "internal test"),
)


def render(evaluation: dict, source: str) -> str:
    items = ", ".join(f"{name} {count:,}" for name, count in evaluation["items_per_set"].items())
    lines = [
        "Macro-F1 by slice, from `" + source + "`, measured on " + evaluation["date"][:10] + ".",
        f"Items: {items}.",
        "",
        "| Detector | " + " | ".join(title for _, title in SLICES) + " | p50 ms | Model MB |",
        "|---|" + "---:|" * len(SLICES) + "---:|---:|",
    ]
    for name, entry in evaluation["detectors"].items():
        cells = " | ".join(fmt(entry["slices"].get(key, {}).get("macro_f1")) for key, _ in SLICES)
        lines.append(
            f"| {name} | {cells} | {fmt(entry['latency']['p50_ms'], 2)} | {entry['model_bytes'] / 1e6:.0f} |"
        )
    lines += [
        "",
        "Per-language precision and recall, the confusion groups, the threshold sweep and the "
        "comparison with the published GlotLID and OpenLID figures are in "
        "[training/RESULTS.md](training/RESULTS.md), generated from the same files.",
    ]
    return "\n".join(lines) + "\n"


def replace_block(text: str, block: str) -> str:
    start, end = text.find(START), text.find(END)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"README.md needs the {START} and {END} markers")
    return text[: start + len(START)] + "\n" + block + text[end:]


def main() -> None:
    path = latest("langid-eval-*.json")
    if path is None:
        raise SystemExit("No results/langid-eval-*.json file to generate from")
    evaluation = json.loads(path.read_text(encoding="utf-8"))
    block = render(evaluation, f"results/{path.name}")
    README.write_text(
        replace_block(README.read_text(encoding="utf-8"), block), encoding="utf-8", newline="\n"
    )
    print(f"Wrote the results block in README.md from results/{path.name}")


if __name__ == "__main__":
    main()
