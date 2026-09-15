import io
import json

import pytest

from tests.lid_fixtures import dakshina_files, sha256_bytes, tar_bytes
from training.data.dakshina import dakshina_rows, extract, fetch_dakshina, read_lexicon
from training.data.rows import Row


def test_extract_keeps_only_romanized_hindi_and_urdu(tmp_path):
    archive = tar_bytes(dakshina_files())
    manifest = extract(io.BytesIO(archive), tmp_path)
    assert sorted(manifest["files"]) == sorted(
        f"{lang}.{suffix}"
        for lang in ("hi", "ur")
        for suffix in (
            "romanized.rejoined.dev.roman.txt",
            "romanized.rejoined.test.roman.txt",
            "translit.sampled.train.tsv",
            "translit.sampled.dev.tsv",
            "translit.sampled.test.tsv",
        )
    )
    assert manifest["archive_sha256"] == sha256_bytes(archive)
    assert manifest["archive_bytes"] == len(archive)
    assert not (tmp_path / "bn.translit.sampled.train.tsv").exists()
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == manifest


def test_extract_names_missing_files(tmp_path):
    files = {k: v for k, v in dakshina_files().items() if "ur.translit.sampled.test" not in k}
    with pytest.raises(FileNotFoundError, match="lacks ur.translit.sampled.test.tsv"):
        extract(io.BytesIO(tar_bytes(files)), tmp_path)


def test_fetch_reuses_an_existing_extraction(tmp_path):
    extract(io.BytesIO(tar_bytes(dakshina_files())), tmp_path)

    def opener(url):
        raise AssertionError("should not download again")

    assert fetch_dakshina(tmp_path, opener=opener)["files"]


def test_lexicon_reads_the_romanization_column(tmp_path):
    path = tmp_path / "lex.tsv"
    path.write_text("किताब\tkitaab\t3\nकिताब\tkitab\t1\nbroken line\n", encoding="utf-8")
    assert read_lexicon(path) == ["kitaab", "kitab"]


def test_rows_and_held_out_sets(tmp_path):
    extract(io.BytesIO(tar_bytes(dakshina_files())), tmp_path)
    rows, test_sentences, test_words = dakshina_rows(tmp_path)
    assert Row("aap kaise hain dev", "hin_Latn", "dakshina", "hi.romanized.rejoined.dev.roman.txt") in rows
    assert Row("kitabdev", "urd_Latn", "dakshina_lexicon", "ur.translit.sampled.dev.tsv") in rows
    assert len(rows) == 2 * (2 + 2 + 1)
    assert test_sentences == {"hin_Latn": ["aap kaise hain test"], "urd_Latn": ["aap kaisay hain test"]}
    assert test_words == {"hin_Latn": ["kitaabtest"], "urd_Latn": ["kitabtest"]}
    assert all("test" not in row.text for row in rows)
