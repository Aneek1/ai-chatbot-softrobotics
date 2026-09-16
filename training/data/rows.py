import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

FIELDS = ("text", "label", "source", "dataset_source", "synthetic")
SYNTHETIC_KINDS = ("opencc", "crop", "code_mixed")


@dataclass(frozen=True)
class Row:
    text: str
    label: str
    source: str  # "openlid", "dakshina" or "dakshina_lexicon"
    dataset_source: str  # OpenLID's dataset_source field, or the Dakshina file it came from
    synthetic: str | None = None  # None for natural text, otherwise one of SYNTHETIC_KINDS


def write_jsonl(rows: Iterable[Row], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[Row]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = [name for name in FIELDS if name not in record]
            if missing:
                raise ValueError(f"{path.name} line {line_number}: missing {', '.join(missing)}")
            if record["synthetic"] is not None and record["synthetic"] not in SYNTHETIC_KINDS:
                raise ValueError(
                    f"{path.name} line {line_number}: unknown synthetic kind {record['synthetic']!r}"
                )
            yield Row(**{name: record[name] for name in FIELDS})
