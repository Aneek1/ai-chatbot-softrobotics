"""Per-run records (spec section 16.5): runs/<run_id>/config.json, metrics.jsonl and final.json."""

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from training.env_report import environment

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"


def new_run_id(name: str, now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    return f"{name}-{now:%Y%m%d-%H%M%S}"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class RunLog:
    def __init__(
        self,
        run_id: str,
        config: dict,
        runs_dir: Path = RUNS_DIR,
        env: dict | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.run_id = run_id
        self.dir = runs_dir / run_id
        # exist_ok=False: a second run with the same id must never overwrite the first one's records.
        self.dir.mkdir(parents=True, exist_ok=False)
        self._clock = clock
        self._started = clock()
        _write_json(
            self.dir / "config.json",
            {
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "config": config,
                "environment": env if env is not None else environment(),
            },
        )

    def elapsed(self) -> float:
        return self._clock() - self._started

    def log_epoch(self, epoch: int, **values) -> dict:
        record = {"epoch": epoch, "elapsed_seconds": round(self.elapsed(), 1), **values}
        with (self.dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def finish(self, **values) -> dict:
        final = {"run_id": self.run_id, "wall_clock_seconds": round(self.elapsed(), 1), **values}
        _write_json(self.dir / "final.json", final)
        return final


def read_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
