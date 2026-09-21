"""Retrieval and answer metrics for the RAG evaluation (spec section 11.2).

citations_all_valid means every [n] in the answer points at a source that was provided. It says
nothing about whether that source supports the sentence; eval/faithfulness.py measures that.
"""

from collections import defaultdict
from collections.abc import Sequence

METRICS = (
    "hit_at_5",
    "hit_at_6",
    "answer_language_correct",
    "has_citation",
    "citations_all_valid",
)


def hit_at_k(retrieved_doc_ids: Sequence[str], expected: Sequence[str], k: int) -> bool:
    wanted = set(expected)
    return any(doc_id in wanted for doc_id in retrieved_doc_ids[:k])


def cited_doc_ids(numbers: Sequence[int], retrieved_doc_ids: Sequence[str]) -> list[str]:
    return [retrieved_doc_ids[n - 1] for n in numbers if 1 <= n <= len(retrieved_doc_ids)]


def rate(values: Sequence[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def summarize(records: Sequence[dict], key: str = "language") -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record[key]].append(record)
    return {
        name: {
            "questions": len(rows),
            **{
                metric: rate([row[metric] for row in rows if row.get(metric) is not None])
                for metric in METRICS
            },
        }
        for name, rows in sorted(groups.items())
    }
