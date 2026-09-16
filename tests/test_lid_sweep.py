import csv

import pytest

from training.sweep import run_grid, summary_row


def test_summary_row_puts_scores_first():
    final = {"run_id": "r1", "wall_clock_seconds": 12.5, "val": {"macro_f1": 0.9, "group_macro_f1": 0.8}}
    assert summary_row({"lr": 0.5}, final) == {
        "run_id": "r1",
        "status": "ok",
        "val_macro_f1": 0.9,
        "val_group_macro_f1": 0.8,
        "wall_clock_seconds": 12.5,
        "lr": 0.5,
    }


def test_failed_run_has_empty_scores():
    row = summary_row({"lr": 1.0}, {"run_id": "r2", "status": "failed", "wall_clock_seconds": 3.0})
    assert row["status"] == "failed"
    assert row["val_macro_f1"] is None


def test_grid_writes_one_csv_row_per_config(tmp_path):
    seen = []

    def train_one(config, run_id):
        seen.append(run_id)
        return {
            "run_id": run_id,
            "wall_clock_seconds": 1.0,
            "val": {"macro_f1": config["lr"], "group_macro_f1": 0.5},
        }

    path = run_grid("demo", [{"lr": 0.1}, {"lr": 0.5}], train_one, tmp_path, make_run_id=lambda name: name)
    assert path == tmp_path / "sweep-demo.csv"
    assert seen == ["demo-00", "demo-01"]
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["run_id"], r["lr"], r["val_macro_f1"]) for r in rows] == [
        ("demo-00", "0.1", "0.1"),
        ("demo-01", "0.5", "0.5"),
    ]


def test_csv_keeps_finished_runs_when_a_later_run_crashes(tmp_path):
    def train_one(config, run_id):
        if config["lr"] > 0.2:
            raise RuntimeError("out of memory")
        return {"run_id": run_id, "wall_clock_seconds": 1.0, "val": {"macro_f1": 0.7, "group_macro_f1": 0.6}}

    with pytest.raises(RuntimeError):
        run_grid("crash", [{"lr": 0.1}, {"lr": 0.5}], train_one, tmp_path, make_run_id=lambda name: name)
    assert "crash-00" in (tmp_path / "sweep-crash.csv").read_text(encoding="utf-8")
