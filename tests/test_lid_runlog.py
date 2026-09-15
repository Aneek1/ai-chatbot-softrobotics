from datetime import UTC, datetime

import pytest

from training.runlog import RunLog, new_run_id, read_json, read_metrics


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_run_id_has_name_and_utc_timestamp():
    assert new_run_id("head-03", datetime(2026, 9, 15, 21, 4, 5, tzinfo=UTC)) == "head-03-20260915-210405"


def test_run_writes_config_metrics_and_final(tmp_path):
    clock = FakeClock()
    run = RunLog("fasttext-00", {"lr": 0.5}, runs_dir=tmp_path, env={"gpu": None}, clock=clock)

    config = read_json(tmp_path / "fasttext-00" / "config.json")
    assert config["config"] == {"lr": 0.5}
    assert config["environment"] == {"gpu": None}

    clock.now = 130.0
    run.log_epoch(1, train_loss=0.9, val_macro_f1=0.8)
    clock.now = 160.04
    run.log_epoch(2, train_loss=0.5, val_macro_f1=0.85)
    final = run.finish(family="fasttext", best_epoch=2)

    assert read_metrics(run.dir) == [
        {"epoch": 1, "elapsed_seconds": 30.0, "train_loss": 0.9, "val_macro_f1": 0.8},
        {"epoch": 2, "elapsed_seconds": 60.0, "train_loss": 0.5, "val_macro_f1": 0.85},
    ]
    assert final == {
        "run_id": "fasttext-00",
        "wall_clock_seconds": 60.0,
        "family": "fasttext",
        "best_epoch": 2,
    }
    assert read_json(run.dir / "final.json") == final


def test_non_latin_text_is_written_as_utf8(tmp_path):
    run = RunLog("r", {"example": "如何制作"}, runs_dir=tmp_path, env={})
    assert "如何制作" in (run.dir / "config.json").read_text(encoding="utf-8")


def test_existing_run_directory_is_never_reused(tmp_path):
    RunLog("same", {}, runs_dir=tmp_path, env={})
    with pytest.raises(FileExistsError):
        RunLog("same", {}, runs_dir=tmp_path, env={})


def test_missing_metrics_file_reads_as_empty(tmp_path):
    assert read_metrics(tmp_path) == []
