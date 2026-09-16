"""fastText supervised specialist (spec section 16.4, model 1).

The full fastText build (fasttext-wheel) and the app's fasttext-predict are both imported as
`fasttext`, so this runs in an isolated environment, like scripts/compress_glotlid.py:

    PYTHONUTF8=1 uv run --no-project --python 3.12 --with fasttext-wheel==0.9.2 --with numpy==1.26.4 \
        python -m training.fasttext_train --sweep

Only the standard library and standard-library-only training modules are imported here.
"""

import argparse
import importlib
import json
import math
import os
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from training.data.rows import Row, read_jsonl
from training.labels import GROUPS, LABELS
from training.metrics import report
from training.runlog import RUNS_DIR, RunLog, new_run_id
from training.sweep import run_grid

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "cache" / "lid"
PREFIX = "__label__"


@dataclass(frozen=True)
class FastTextConfig:
    epoch: int
    lr: float
    dim: int
    minn: int
    maxn: int
    word_ngrams: int = 1
    bucket: int = 2_000_000
    loss: str = "softmax"
    include_synthetic: bool = True
    seed: int = 20260915


# Six of the 36 combinations in spec section 16.4, chosen so every listed value appears at least once.
SWEEP = (
    FastTextConfig(epoch=5, lr=0.5, dim=16, minn=1, maxn=4),
    FastTextConfig(epoch=10, lr=0.5, dim=16, minn=2, maxn=5),
    FastTextConfig(epoch=10, lr=0.5, dim=64, minn=2, maxn=5),
    FastTextConfig(epoch=10, lr=1.0, dim=64, minn=1, maxn=4),
    FastTextConfig(epoch=25, lr=0.1, dim=64, minn=2, maxn=5),
    FastTextConfig(epoch=25, lr=0.5, dim=16, minn=1, maxn=4),
)


def to_line(row: Row) -> str:
    return f"{PREFIX}{row.label} {row.text}"


def write_training_file(rows, path: Path, include_synthetic: bool) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            if include_synthetic or row.synthetic is None:
                handle.write(to_line(row) + "\n")
                count += 1
    return count


def evaluate(model, rows: list[Row]) -> tuple[dict, float]:
    predictions, loss = [], 0.0
    for row in rows:
        labels, probabilities = model.predict(row.text, k=-1)
        predictions.append(labels[0].removeprefix(PREFIX))
        by_label = dict(zip(labels, probabilities, strict=True))
        loss -= math.log(max(float(by_label.get(PREFIX + row.label, 0.0)), 1e-9))
    return report([r.label for r in rows], predictions, LABELS, GROUPS), loss / max(len(rows), 1)


def train_one(
    config: FastTextConfig,
    data_dir: Path,
    runs_dir: Path,
    run_id: str,
    fasttext_module=None,
    env: dict | None = None,
    threads: int | None = None,
) -> dict:
    fasttext = fasttext_module or importlib.import_module("fasttext")
    # fastText's SGD is asynchronous across threads, so the seed only makes a run repeatable with
    # threads=1. The default uses every thread for speed; config.json records which case applies.
    threads = threads or os.cpu_count() or 1
    run = RunLog(
        run_id,
        {
            "family": "fasttext",
            **asdict(config),
            "threads": threads,
            "reproducible": threads == 1,
            "data_dir": str(data_dir),
        },
        runs_dir,
        env,
    )
    train_file = run.dir / "train.txt"
    lines = write_training_file(read_jsonl(data_dir / "train.jsonl"), train_file, config.include_synthetic)
    val_rows = list(read_jsonl(data_dir / "val_select.jsonl"))
    try:
        model = fasttext.train_supervised(
            input=str(train_file),
            epoch=config.epoch,
            lr=config.lr,
            dim=config.dim,
            minn=config.minn,
            maxn=config.maxn,
            wordNgrams=config.word_ngrams,
            bucket=config.bucket,
            loss=config.loss,
            thread=threads,
            seed=config.seed,
            verbose=0,
        )
    except RuntimeError as error:  # fastText raises "Encountered NaN." when the learning rate is too high
        return run.finish(family="fasttext", status="failed", error=str(error), train_lines=lines)
    finally:
        train_file.unlink(missing_ok=True)
    val, val_loss = evaluate(model, val_rows)
    run.log_epoch(
        config.epoch,
        train_loss=None,
        val_loss=round(val_loss, 4),
        val_macro_f1=val["macro_f1"],
        val_group_macro_f1=val["group_macro_f1"],
        val_group_f1=val["group_f1"],
    )
    model_path = run.dir / "model.bin"
    model.save_model(str(model_path))
    return run.finish(
        family="fasttext",
        status="ok",
        train_lines=lines,
        best_epoch=config.epoch,
        val=val,
        val_loss=round(val_loss, 4),
        model_file=model_path.name,
        model_bytes=model_path.stat().st_size,
        note="fastText's Python API reports no training loss; one metrics line follows the last epoch.",
    )


def config_of_run(run_dir: Path) -> FastTextConfig:
    stored = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["config"]
    return FastTextConfig(**{f.name: stored[f.name] for f in fields(FastTextConfig)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--runs", type=Path, default=RUNS_DIR)
    parser.add_argument("--sweep", action="store_true", help="train every configuration in SWEEP")
    parser.add_argument(
        "--without-synthetic", metavar="RUN_ID", help="retrain that run's configuration on natural rows only"
    )
    parser.add_argument(
        "--threads", type=int, help="training threads (default: all); 1 makes a run repeatable with its seed"
    )
    args = parser.parse_args()
    if args.sweep:
        path = run_grid(
            "fasttext",
            [asdict(c) for c in SWEEP],
            lambda cfg, run_id: train_one(
                FastTextConfig(**cfg), args.data, args.runs, run_id, threads=args.threads
            ),
            args.runs,
        )
        print(f"Wrote {path}")
    if args.without_synthetic:
        config = replace(config_of_run(args.runs / args.without_synthetic), include_synthetic=False)
        final = train_one(config, args.data, args.runs, new_run_id("fasttext-natural"), threads=args.threads)
        print(f"{final['run_id']}: group macro-F1 {final.get('val', {}).get('group_macro_f1')}")


if __name__ == "__main__":
    main()
