import hashlib
import urllib.request
from collections.abc import Callable
from pathlib import Path

CHUNK = 1 << 20


class ChecksumMismatch(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, target: Path, sha256: str, opener: Callable = urllib.request.urlopen) -> Path:
    """Download once into target, verified against a pinned SHA-256."""
    if target.exists():
        actual = sha256_file(target)
        if actual == sha256:
            return target
        raise ChecksumMismatch(
            f"{target} exists but has SHA-256 {actual}, expected {sha256}. Delete it and run again."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    with opener(url) as response, partial.open("wb") as handle:
        while block := response.read(CHUNK):
            digest.update(block)
            handle.write(block)
    if digest.hexdigest() != sha256:
        partial.unlink()
        raise ChecksumMismatch(f"{url} has SHA-256 {digest.hexdigest()}, expected {sha256}")
    partial.replace(target)
    return target
