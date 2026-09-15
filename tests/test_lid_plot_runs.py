import pytest

pytest.importorskip("matplotlib")

from training.plot_runs import plot_run  # noqa: E402
from training.runlog import RunLog  # noqa: E402


def test_plot_writes_a_png_next_to_the_metrics(tmp_path):
    run = RunLog("head-00", {}, runs_dir=tmp_path, env={})
    for epoch, f1 in ((1, 0.6), (2, 0.7), (3, 0.72)):
        run.log_epoch(
            epoch,
            train_loss=1 / epoch,
            val_loss=1.2 / epoch,
            val_macro_f1=f1,
            val_group_f1={"han": f1 - 0.1, "malay_indonesian": f1 - 0.2},
        )
    path = plot_run(run.dir)
    assert path == run.dir / "curves.png"
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_fasttext_run_without_training_loss_still_plots(tmp_path):
    run = RunLog("fasttext-00", {}, runs_dir=tmp_path, env={})
    run.log_epoch(5, train_loss=None, val_loss=0.4, val_macro_f1=0.8, val_group_f1={"han": 0.7})
    assert plot_run(run.dir).exists()


def test_run_without_metrics_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="has no metrics.jsonl lines"):
        plot_run(tmp_path)
