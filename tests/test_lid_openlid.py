import io
import json

import pytest

from training.data.clean import Cleaner, held_out_keys
from training.data.openlid import (
    OpenLIDSampler,
    iter_parquet,
    language_names,
    list_shards,
    sample_shards,
    shard_order,
    shard_paths,
)
from training.labels import OPENLID_LABELS

TREE = [
    {"type": "file", "path": "data/train-00001-of-00044-b.parquet"},
    {"type": "file", "path": "data/train-00000-of-00044-a.parquet"},
    {"type": "directory", "path": "data/sub"},
    {"type": "file", "path": "data/README.md"},
]


def test_shard_paths_keeps_sorted_parquet_files():
    assert shard_paths(TREE) == ["data/train-00000-of-00044-a.parquet", "data/train-00001-of-00044-b.parquet"]


def test_list_shards_reads_the_tree_api():
    assert list_shards(opener=lambda url: io.BytesIO(json.dumps(TREE).encode())) == shard_paths(TREE)


def test_shard_order_is_seeded():
    paths = [f"s{i}" for i in range(10)]
    assert shard_order(paths, 5) == shard_order(paths, 5)
    assert sorted(shard_order(paths, 5)) == paths
    assert shard_order(paths, 5) != paths


def test_language_names_from_schema_metadata():
    metadata = {
        b"huggingface": json.dumps(
            {"info": {"features": {"language": {"names": ["plt_Latn", "eng_Latn"]}}}}
        ).encode()
    }
    assert language_names(metadata) == ["plt_Latn", "eng_Latn"]


def test_sampler_caps_targets_and_other_languages_separately():
    sampler = OpenLIDSampler(Cleaner(), target_cap=2, other_cap=1)
    for i in range(4):
        sampler.offer("zsm_Latn", f"ayat {i}", "leipzig")
        sampler.offer("fra_Latn", f"phrase {i}", "lti")
        sampler.offer("deu_Latn", f"satz {i}", "lti")
    assert sampler.counts == {"zsm_Latn": 2, "fra_Latn": 1, "deu_Latn": 1}
    assert {row.label for row in sampler.rows} == {"zsm_Latn", "other"}


def test_rejected_lines_do_not_use_up_the_cap():
    sampler = OpenLIDSampler(Cleaner(held_out_keys(["FLORES line"])), target_cap=1, other_cap=1)
    sampler.offer("eng_Latn", "FLORES line", "lti")
    sampler.offer("eng_Latn", "x", "lti")
    sampler.offer("eng_Latn", "a real line", "lti")
    assert [row.text for row in sampler.rows] == ["a real line"]


def test_sampling_stops_once_every_target_is_full():
    sampler = OpenLIDSampler(Cleaner(), target_cap=1, other_cap=1)
    full_shard = [(language, f"{language} text", "lti") for language in sorted(OPENLID_LABELS)]
    never_read = [("eng_Latn", "should not be read", "lti")]
    used = sample_shards([full_shard, never_read], sampler, log=lambda message: None)
    assert used == 1
    assert sampler.targets_full()


def test_reads_parquet_with_integer_labels(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    names = ["plt_Latn", "eng_Latn"]
    table = pa.table(
        {"text": ["Manao ahoana", "Hello"], "language": [0, 1], "dataset_source": ["mt560", "lti"]}
    )
    metadata = {b"huggingface": json.dumps({"info": {"features": {"language": {"names": names}}}}).encode()}
    table = table.replace_schema_metadata(metadata)
    path = tmp_path / "shard.parquet"
    pq.write_table(table, path)
    assert list(iter_parquet(path)) == [("plt_Latn", "Manao ahoana", "mt560"), ("eng_Latn", "Hello", "lti")]
