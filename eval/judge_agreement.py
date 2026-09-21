"""How often the citation judge agrees with a person (spec section 11.2).

    PYTHONUTF8=1 uv run python -m eval.judge_agreement --draft
    (fill in the "human" field of every row with supported, unsupported or unclear)
    PYTHONUTF8=1 uv run python -m eval.judge_agreement --score

--draft writes a stratified sample of judged pairs with an empty human field. --score compares the
two columns and writes the agreement into the results file, so the faithfulness numbers are always
published next to the evidence for how much the judge can be trusted.
"""

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from eval.faithfulness import VERDICTS
from training.results_table import latest

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
SAMPLE_PATH = REPO_ROOT / "eval" / "faithfulness-sample.jsonl"
SAMPLE_FIELDS = ("question_id", "language", "chunk_id", "sentence", "chunk_text", "judge", "human")
DEFAULT_COUNT = 30
SEED = 20260916


class SampleError(ValueError):
    """A hand-labelled sample the score pass cannot use."""


def draft_sample(results: dict, count: int = DEFAULT_COUNT, rng: random.Random | None = None) -> list[dict]:
    """Up to count/3 pairs of each verdict, so a rare verdict is not missed by chance."""
    rng = rng or random.Random(SEED)
    by_verdict: dict[str, list[dict]] = defaultdict(list)
    for record in results["judged"]:
        by_verdict[record["verdict"]].append(record)
    per_verdict = max(1, count // len(VERDICTS))
    sample = []
    for verdict in VERDICTS:
        rows = by_verdict.get(verdict, [])
        for record in rng.sample(rows, min(per_verdict, len(rows))):
            sample.append(
                {
                    "question_id": record["question_id"],
                    "language": record["language"],
                    "chunk_id": record["chunk_id"],
                    "sentence": record["sentence"],
                    "chunk_text": record["chunk_text"],
                    "judge": record["verdict"],
                    "human": "",
                }
            )
    return sample


def write_sample(rows: Sequence[dict], path: Path = SAMPLE_PATH) -> int:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def read_sample(path: Path = SAMPLE_PATH) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def agreement(sample: Sequence[dict]) -> dict:
    for row in sample:
        if row["human"] not in ("", *VERDICTS):
            raise SampleError(f"{row['human']!r} is not one of {', '.join(VERDICTS)} or empty")
    labelled = [row for row in sample if row["human"]]
    total = len(labelled)
    return {
        "pairs": len(sample),
        "labelled": total,
        "agreement": round(sum(r["judge"] == r["human"] for r in labelled) / total, 4) if total else None,
        "judge_supported_rate": round(
            sum(r["judge"] == "supported" for r in labelled) / total, 4
        )
        if total
        else None,
        "human_supported_rate": round(
            sum(r["human"] == "supported" for r in labelled) / total, 4
        )
        if total
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", action="store_true", help="write an unlabelled sample")
    parser.add_argument("--score", action="store_true", help="score the labelled sample")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()
    path = args.results or latest("rag-eval-*.json", RESULTS_DIR)
    if path is None:
        raise SystemExit("No results/rag-eval-*.json file; run eval.rag_eval first")
    results = json.loads(path.read_text(encoding="utf-8"))

    if args.draft:
        written = write_sample(draft_sample(results, args.count))
        print(f"Wrote {written} pairs to {SAMPLE_PATH.relative_to(REPO_ROOT)}; fill in every 'human' field")
        return
    if not args.score:
        raise SystemExit("Pass --draft or --score")
    scores = agreement(read_sample())
    results["judge_agreement"] = {"sample_file": "eval/faithfulness-sample.jsonl", **scores}
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"{scores['labelled']} labelled pairs, agreement {scores['agreement']}")
    print(
        f"judge supported {scores['judge_supported_rate']}, "
        f"person supported {scores['human_supported_rate']}"
    )
    print(f"Wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
