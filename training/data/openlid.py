"""OpenLID sampling (spec section 16.3): shard by shard, capped per language, so the 16.6 GB dataset
never has to be downloaded in full."""

import json
import random
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from training.data.clean import Cleaner
from training.data.rows import Row
from training.labels import OPENLID_LABELS, OTHER, training_label

OPENLID_REPO = "laurievb/open-lid-dataset"
OPENLID_REVISION = "67b9a87a9f52285a666aea50a0e118b8db8f6daa"
OPENLID_SOURCE = {
    "name": "OpenLID",
    "url": f"https://huggingface.co/datasets/{OPENLID_REPO}",
    "revision": OPENLID_REVISION,
    "licence": "other: varies by source dataset (see the dataset card); the data is not redistributed",
}
TREE_URL = f"https://huggingface.co/api/datasets/{OPENLID_REPO}/tree/{OPENLID_REVISION}/data"


def shard_paths(tree: list[dict]) -> list[str]:
    return sorted(
        entry["path"] for entry in tree if entry.get("type") == "file" and entry["path"].endswith(".parquet")
    )


def list_shards(opener: Callable = urllib.request.urlopen) -> list[str]:
    with opener(TREE_URL) as response:
        return shard_paths(json.load(response))


def shard_order(paths: list[str], seed: int) -> list[str]:
    ordered = list(paths)
    random.Random(seed).shuffle(ordered)
    return ordered


def download_shard(path: str, cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            OPENLID_REPO,
            path,
            repo_type="dataset",
            revision=OPENLID_REVISION,
            local_dir=cache_dir / "openlid",
        )
    )


def language_names(metadata: dict[bytes, bytes]) -> list[str]:
    """The language column is an integer class label; its names are in the parquet schema metadata."""
    return json.loads(metadata[b"huggingface"])["info"]["features"]["language"]["names"]


def iter_parquet(path: Path, batch_size: int = 100_000) -> Iterator[tuple[str, str, str]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    names = language_names(parquet.schema_arrow.metadata)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=["text", "language", "dataset_source"]):
        texts = batch.column("text").to_pylist()
        languages = batch.column("language").to_pylist()
        sources = batch.column("dataset_source").to_pylist()
        for text, language, source in zip(texts, languages, sources, strict=True):
            yield (names[language] if isinstance(language, int) else language), text, source


class OpenLIDSampler:
    def __init__(self, cleaner: Cleaner, target_cap: int, other_cap: int):
        self._cleaner = cleaner
        self.target_cap = target_cap
        self.other_cap = other_cap
        self.counts: Counter = Counter()  # rows kept per OpenLID language
        self.rows: list[Row] = []

    def offer(self, language: str, text: str, dataset_source: str) -> None:
        label = training_label(language)
        cap = self.other_cap if label == OTHER else self.target_cap
        if self.counts[language] >= cap:
            return
        row = self._cleaner.accept(Row(text, label, "openlid", dataset_source))
        if row is not None:
            self.counts[language] += 1
            self.rows.append(row)

    def targets_full(self) -> bool:
        return all(self.counts[language] >= self.target_cap for language in OPENLID_LABELS)


def sample_shards(
    shards: Iterable[Iterable[tuple[str, str, str]]],
    sampler: OpenLIDSampler,
    log: Callable[[str], None] = print,
) -> int:
    used = 0
    for shard in shards:
        for language, text, source in shard:
            sampler.offer(language, text, source)
        used += 1
        short = {
            lang: sampler.counts[lang]
            for lang in sorted(OPENLID_LABELS)
            if sampler.counts[lang] < sampler.target_cap
        }
        log(f"shard {used}: {len(sampler.rows)} rows kept; below cap: {short or 'none'}")
        if sampler.targets_full():
            break
    return used
