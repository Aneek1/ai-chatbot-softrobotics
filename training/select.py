"""Choose the specialist (spec section 16.6) on validation macro-F1 over the confusion-group labels.

Usage:
    uv run python -m training.select

Ties are broken by macro-F1 over all labels, then by the smaller model file. The test split and the
held-out sets play no part here; eval/ scores the chosen models once, afterwards.
"""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from training.runlog import RUNS_DIR, read_json

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FAMILIES = ("fasttext", "e5_head", "e5_finetune")
SUMMARY_FILES = ("config.json", "metrics.jsonl", "final.json")


def load_finals(runs_dir: Path) -> list[dict]:
    finals = []
    for path in sorted(runs_dir.glob("*/final.json")):
        final = read_json(path)
        if final.get("status") == "ok" and final.get("val") and final.get("family") in FAMILIES:
            finals.append(final)
    return finals


def rank_key(final: dict) -> tuple[float, float, int]:
    return final["val"]["group_macro_f1"], final["val"]["macro_f1"], -final["model_bytes"]


def choose(finals: list[dict]) -> dict:
    if not finals:
        raise ValueError("no finished runs to choose from")
    ranked = sorted(finals, key=rank_key, reverse=True)
    per_family = {
        family: next((f["run_id"] for f in ranked if f["family"] == family), None) for family in FAMILIES
    }
    return {
        "selection_metric": "validation macro-F1 over confusion-group labels (val_select)",
        "overall": ranked[0]["run_id"],
        "overall_family": ranked[0]["family"],
        "per_family": per_family,
        "ranking": [
            {
                "run_id": f["run_id"],
                "family": f["family"],
                "val_group_macro_f1": f["val"]["group_macro_f1"],
                "val_macro_f1": f["val"]["macro_f1"],
                "val_group_f1": f["val"]["group_f1"],
                "model_bytes": f["model_bytes"],
                "wall_clock_seconds": f["wall_clock_seconds"],
            }
            for f in ranked
        ],
    }


def copy_summaries(run_ids: list[str], runs_dir: Path, results_dir: Path) -> list[Path]:
    """Copy the small text records of chosen runs into results/ (runs/ itself is git-ignored)."""
    copied = []
    for run_id in run_ids:
        target = results_dir / "lid-runs" / run_id
        target.mkdir(parents=True, exist_ok=True)
        for name in SUMMARY_FILES:
            if (runs_dir / run_id / name).exists():
                copied.append(Path(shutil.copy2(runs_dir / run_id / name, target / name)))
    return copied


def copy_sweeps(runs_dir: Path, results_dir: Path) -> list[Path]:
    target = results_dir / "lid-runs"
    target.mkdir(parents=True, exist_ok=True)
    return [Path(shutil.copy2(path, target / path.name)) for path in sorted(runs_dir.glob("sweep-*.csv"))]


def main() -> None:
    selection = choose(load_finals(RUNS_DIR))
    now = datetime.now(UTC)
    selection["selected_at"] = now.isoformat(timespec="seconds")
    kept = sorted(
        {run_id for run_id in selection["per_family"].values() if run_id}
        | {r["run_id"] for r in selection["ranking"] if r["run_id"].startswith("fasttext-natural")}
    )
    copy_summaries(kept, RUNS_DIR, RESULTS_DIR)
    copy_sweeps(RUNS_DIR, RESULTS_DIR)
    path = RESULTS_DIR / f"lid-selection-{now:%Y-%m-%d}.json"
    path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(
        f"Chosen: {selection['overall']} ({selection['overall_family']}); wrote {path.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
