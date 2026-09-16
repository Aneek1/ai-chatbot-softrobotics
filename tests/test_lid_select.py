import json

import pytest

from training.select import choose, copy_summaries, copy_sweeps, load_finals


def final(run_id, family, group_f1, macro_f1, size, status="ok"):
    return {
        "run_id": run_id,
        "family": family,
        "status": status,
        "model_bytes": size,
        "wall_clock_seconds": 10.0,
        "val": {"group_macro_f1": group_f1, "macro_f1": macro_f1, "group_f1": {"han": group_f1}},
    }


def write_run(runs_dir, record):
    run_dir = runs_dir / record["run_id"]
    run_dir.mkdir(parents=True)
    (run_dir / "final.json").write_text(json.dumps(record), encoding="utf-8")
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "model.pt").write_bytes(b"weights")


def test_group_f1_decides_then_macro_f1_then_size():
    finals = [
        final("ft-0", "fasttext", 0.90, 0.95, 100),
        final("head-0", "e5_head", 0.92, 0.90, 500),
        final("head-1", "e5_head", 0.92, 0.90, 400),
        final("tune-0", "e5_finetune", 0.91, 0.99, 900),
    ]
    selection = choose(finals)
    assert selection["overall"] == "head-1"
    assert selection["per_family"] == {"fasttext": "ft-0", "e5_head": "head-1", "e5_finetune": "tune-0"}
    assert [r["run_id"] for r in selection["ranking"]] == ["head-1", "head-0", "tune-0", "ft-0"]


def test_missing_family_is_none():
    assert choose([final("ft-0", "fasttext", 0.9, 0.9, 1)])["per_family"]["e5_finetune"] is None


def test_no_runs_is_an_error():
    with pytest.raises(ValueError, match="no finished runs"):
        choose([])


def test_failed_and_unknown_runs_are_not_candidates(tmp_path):
    write_run(tmp_path, final("ft-0", "fasttext", 0.9, 0.9, 1))
    write_run(tmp_path, final("ft-1", "fasttext", 0.99, 0.99, 1, status="failed"))
    write_run(tmp_path, final("x-0", "mystery", 0.99, 0.99, 1))
    assert [f["run_id"] for f in load_finals(tmp_path)] == ["ft-0"]


def test_only_text_summaries_are_copied(tmp_path):
    runs, results = tmp_path / "runs", tmp_path / "results"
    write_run(runs, final("head-1", "e5_head", 0.9, 0.9, 1))
    copied = copy_summaries(["head-1"], runs, results)
    assert sorted(p.name for p in copied) == ["config.json", "final.json", "metrics.jsonl"]
    assert not (results / "lid-runs" / "head-1" / "model.pt").exists()


def test_sweep_summaries_are_copied(tmp_path):
    runs, results = tmp_path / "runs", tmp_path / "results"
    runs.mkdir()
    (runs / "sweep-fasttext.csv").write_text("run_id\nft-0\n", encoding="utf-8")
    copied = copy_sweeps(runs, results)
    assert [p.name for p in copied] == ["sweep-fasttext.csv"]
    assert (results / "lid-runs" / "sweep-fasttext.csv").read_text(encoding="utf-8") == "run_id\nft-0\n"
