"""Dakshina v1.0 (CC BY-SA 4.0): romanized Hindi and Urdu.

The archive is 2.0 GB and has no index, so it is streamed once and only the romanized sentence files
and word lexicons for Hindi and Urdu are kept. The test files are held out: they are never trained on.
"""

import hashlib
import json
import tarfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

from training.data.rows import Row
from training.labels import DAKSHINA_LABELS

DAKSHINA_URL = "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"
DAKSHINA_SOURCE = {
    "name": "Dakshina v1.0",
    "url": DAKSHINA_URL,
    "bytes": 2_008_340_480,
    "licence": "CC-BY-SA-4.0",
    "citation": "Roark et al., Processing South Asian Languages Written in the Latin Script: "
    "the Dakshina Dataset, LREC 2020",
}
CHUNK = 1 << 20


def wanted_files(language: str) -> list[str]:
    return [
        f"{language}/romanized/{language}.romanized.rejoined.dev.roman.txt",
        f"{language}/romanized/{language}.romanized.rejoined.test.roman.txt",
        f"{language}/lexicons/{language}.translit.sampled.train.tsv",
        f"{language}/lexicons/{language}.translit.sampled.dev.tsv",
        f"{language}/lexicons/{language}.translit.sampled.test.tsv",
    ]


class _HashingReader:
    def __init__(self, raw: BinaryIO):
        self._raw = raw
        self._digest = hashlib.sha256()
        self.bytes = 0

    def read(self, size: int = -1) -> bytes:
        data = self._raw.read(size)
        self._digest.update(data)
        self.bytes += len(data)
        return data

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def extract(stream: BinaryIO, target_dir: Path, languages: tuple[str, ...] = tuple(DAKSHINA_LABELS)) -> dict:
    wanted = {path for language in languages for path in wanted_files(language)}
    target_dir.mkdir(parents=True, exist_ok=True)
    reader = _HashingReader(stream)
    files = {}
    with tarfile.open(fileobj=reader, mode="r|") as tar:
        for member in tar:
            inner = member.name.split("/", 1)[1] if "/" in member.name else member.name
            if member.isfile() and inner in wanted:
                data = tar.extractfile(member).read()
                name = Path(inner).name
                (target_dir / name).write_bytes(data)
                files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    while reader.read(CHUNK):
        pass  # hash the archive to its last byte so the manifest records the whole file
    missing = sorted(Path(path).name for path in wanted if Path(path).name not in files)
    if missing:
        raise FileNotFoundError(f"Dakshina archive lacks {', '.join(missing)}")
    manifest = {
        "archive_sha256": reader.hexdigest(),
        "archive_bytes": reader.bytes,
        "files": dict(sorted(files.items())),
    }
    (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def fetch_dakshina(target_dir: Path, opener: Callable = urllib.request.urlopen) -> dict:
    manifest_path = target_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all((target_dir / name).exists() for name in manifest["files"]):
            return manifest
    with opener(DAKSHINA_URL) as response:
        return extract(response, target_dir)


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_lexicon(path: Path) -> list[str]:
    """Romanizations (second column) in file order."""
    return [line.split("\t")[1] for line in read_lines(path) if line.count("\t") >= 2]


def dakshina_rows(target_dir: Path) -> tuple[list[Row], dict[str, list[str]], dict[str, list[str]]]:
    """Returns training rows (dev sentences, train and dev lexicons) and the held-out
    test sentences and test words per label."""
    rows: list[Row] = []
    test_sentences: dict[str, list[str]] = {}
    test_words: dict[str, list[str]] = {}
    for language, label in DAKSHINA_LABELS.items():
        dev = f"{language}.romanized.rejoined.dev.roman.txt"
        rows += [Row(text, label, "dakshina", dev) for text in read_lines(target_dir / dev)]
        for split in ("train", "dev"):
            name = f"{language}.translit.sampled.{split}.tsv"
            rows += [Row(word, label, "dakshina_lexicon", name) for word in read_lexicon(target_dir / name)]
        test_sentences[label] = read_lines(target_dir / f"{language}.romanized.rejoined.test.roman.txt")
        test_words[label] = read_lexicon(target_dir / f"{language}.translit.sampled.test.tsv")
    return rows, test_sentences, test_words
