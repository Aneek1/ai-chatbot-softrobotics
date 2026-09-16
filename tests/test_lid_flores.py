import io

import pytest

from tests.lid_fixtures import FLORES_FILES, sha256_bytes, tar_bytes, write_flores_archive
from training.data.download import ChecksumMismatch, fetch
from training.data.flores import FLORES_SHA256, FLORES_SOURCE, read_flores


def opener_for(data: bytes):
    calls = []

    def opener(url):
        calls.append(url)
        return io.BytesIO(data)

    opener.calls = calls
    return opener


def test_fetch_verifies_and_caches(tmp_path):
    data = b"archive bytes"
    opener = opener_for(data)
    target = tmp_path / "cache" / "file.tar.gz"
    assert fetch("https://example.invalid/f", target, sha256_bytes(data), opener=opener) == target
    assert target.read_bytes() == data
    fetch("https://example.invalid/f", target, sha256_bytes(data), opener=opener)
    assert len(opener.calls) == 1


def test_fetch_rejects_a_wrong_checksum_and_leaves_nothing(tmp_path):
    target = tmp_path / "file.tar.gz"
    with pytest.raises(ChecksumMismatch, match="expected 00"):
        fetch("https://example.invalid/f", target, "00", opener=opener_for(b"tampered"))
    assert list(tmp_path.iterdir()) == []


def test_fetch_refuses_an_existing_file_with_the_wrong_checksum(tmp_path):
    target = tmp_path / "file.tar.gz"
    target.write_bytes(b"old")
    with pytest.raises(ChecksumMismatch, match="Delete it and run again"):
        fetch("https://example.invalid/f", target, sha256_bytes(b"new"), opener=opener_for(b"new"))


def test_pinned_release_facts():
    assert FLORES_SHA256 == "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"
    assert FLORES_SOURCE["bytes"] == 25_585_843
    assert FLORES_SOURCE["licence"] == "CC-BY-SA-4.0"


def test_reads_one_split_by_language_code(tmp_path):
    archive = write_flores_archive(tmp_path / "flores.tar.gz")
    devtest = read_flores(archive, "devtest")
    assert list(devtest) == ["eng_Latn", "fra_Latn", "zho_Hant", "zsm_Latn"]
    assert devtest["zho_Hant"] == ["如何製作軟氣動執行器"]
    assert read_flores(archive, "dev")["eng_Latn"] == ["Dev sentence one.", "Dev sentence two."]


def test_unknown_split_is_rejected(tmp_path):
    archive = tmp_path / "flores.tar.gz"
    archive.write_bytes(tar_bytes(FLORES_FILES, "gz"))
    with pytest.raises(ValueError, match="unknown FLORES split 'test'"):
        read_flores(archive, "test")
