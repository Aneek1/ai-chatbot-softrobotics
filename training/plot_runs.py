"""Training curves from runs/<run_id>/metrics.jsonl (spec section 16.5).

Usage:
    uv run --group train python -m training.plot_runs runs/<run_id> [runs/<run_id> ...]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from training.runlog import read_metrics  # noqa: E402


def plot_run(run_dir: Path) -> Path:
    metrics = read_metrics(run_dir)
    if not metrics:
        raise ValueError(f"{run_dir} has no metrics.jsonl lines")
    epochs = [m["epoch"] for m in metrics]
    figure, (loss_axis, f1_axis) = plt.subplots(1, 2, figsize=(10, 4))
    for key, name in (("train_loss", "train"), ("val_loss", "validation")):
        points = [(m["epoch"], m[key]) for m in metrics if m.get(key) is not None]
        if points:
            loss_axis.plot(*zip(*points, strict=True), marker="o", label=name)
    loss_axis.set_xlabel("epoch")
    loss_axis.set_ylabel("cross-entropy")
    if loss_axis.lines:
        loss_axis.legend()
    f1_axis.plot(epochs, [m["val_macro_f1"] for m in metrics], marker="o", label="macro-F1, all labels")
    for group in metrics[0].get("val_group_f1", {}):
        f1_axis.plot(epochs, [m["val_group_f1"][group] for m in metrics], marker=".", label=group)
    f1_axis.set_xlabel("epoch")
    f1_axis.set_ylabel("validation F1")
    f1_axis.set_ylim(0, 1)
    f1_axis.legend(fontsize=8)
    figure.suptitle(run_dir.name)
    figure.tight_layout()
    out = run_dir / "curves.png"
    figure.savefig(out, dpi=120)
    plt.close(figure)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    for run_dir in parser.parse_args().runs:
        print(plot_run(run_dir))


if __name__ == "__main__":
    main()
