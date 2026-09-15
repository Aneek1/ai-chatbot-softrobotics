"""FLORES-200 from Meta's original release (no account needed), pinned by SHA-256.

It is used only for evaluation and for the contamination check; no FLORES sentence may enter training.
"""

import tarfile
from collections.abc import Callable
from pathlib import Path

from training.data.download import fetch

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "cache"

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
FLORES_SHA256 = "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"
FLORES_SOURCE = {
    "name": "FLORES-200",
    "url": FLORES_URL,
    "sha256": FLORES_SHA256,
    "bytes": 25_585_843,
    "last_modified": "2022-07-14",
    "licence": "CC-BY-SA-4.0",
    "licence_source": "https://github.com/facebookresearch/flores/blob/main/flores200/README.md",
}
SPLITS = ("dev", "devtest")


def fetch_flores(cache_dir: Path = CACHE_DIR, opener: Callable | None = None) -> Path:
    target = cache_dir / "flores200_dataset.tar.gz"
    if opener is None:
        return fetch(FLORES_URL, target, FLORES_SHA256)
    return fetch(FLORES_URL, target, FLORES_SHA256, opener=opener)


def read_flores(archive: Path, split: str) -> dict[str, list[str]]:
    """FLORES code (e.g. zho_Hant) -> sentences in file order."""
    if split not in SPLITS:
        raise ValueError(f"unknown FLORES split {split!r}; expected one of {SPLITS}")
    sentences: dict[str, list[str]] = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            parts = member.name.split("/")
            if member.isfile() and len(parts) >= 2 and parts[-2] == split and parts[-1].endswith(f".{split}"):
                code = parts[-1].removesuffix(f".{split}")
                sentences[code] = tar.extractfile(member).read().decode("utf-8").splitlines()
    return dict(sorted(sentences.items()))
