"""Run a grid of configurations and summarise it in runs/sweep-<name>.csv (spec section 16.5).

Standard library only: the fastText sweep imports it from an isolated environment.
"""

import csv
from collections.abc import Callable, Sequence
from pathlib import Path

from training.runlog import new_run_id

SUMMARY_FIELDS = ("run_id", "status", "val_macro_f1", "val_group_macro_f1", "wall_clock_seconds")


def summary_row(config: dict, final: dict) -> dict:
    val = final.get("val") or {}
    return {
        "run_id": final["run_id"],
        "status": final.get("status", "ok"),
        "val_macro_f1": val.get("macro_f1"),
        "val_group_macro_f1": val.get("group_macro_f1"),
        "wall_clock_seconds": final.get("wall_clock_seconds"),
        **config,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(SUMMARY_FIELDS)
    for row in rows:
        fields += [key for key in row if key not in fields]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_grid(
    name: str,
    configs: Sequence[dict],
    train_one: Callable[[dict, str], dict],
    runs_dir: Path,
    make_run_id: Callable[[str], str] = new_run_id,
) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"sweep-{name}.csv"
    rows: list[dict] = []
    for index, config in enumerate(configs):
        final = train_one(config, make_run_id(f"{name}-{index:02d}"))
        rows.append(summary_row(config, final))
        write_csv(path, rows)  # rewritten after each run, so an interrupted sweep keeps what finished
        print(f"{final['run_id']}: group macro-F1 {rows[-1]['val_group_macro_f1']}")
    return path
