"""Download model files into a flat models/ folder and record where they came from.

Usage:
    uv run python scripts/download_models.py            # e5-small only
    uv run python scripts/download_models.py --glotlid  # also the 1.69 GB GlotLID model.bin
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import hf_hub_download, model_info

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"

E5_REPO = "intfloat/multilingual-e5-small"
E5_FILES = [
    "onnx/model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
]
GLOTLID_REPO = "cis-lmu/glotlid"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(repo: str, filenames: list[str], target: Path) -> dict:
    revision = model_info(repo).sha
    files = {}
    for name in filenames:
        path = Path(hf_hub_download(repo, name, revision=revision, local_dir=target))
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        print(f"  {repo}/{name} -> {path.relative_to(REPO_ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")
    return {"repo": repo, "revision": revision, "files": files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glotlid", action="store_true", help="also download the full GlotLID model.bin")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    manifest = {"downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"), "models": []}
    manifest["models"].append(fetch(E5_REPO, E5_FILES, MODELS_DIR / "multilingual-e5-small"))
    if args.glotlid:
        entry = fetch(GLOTLID_REPO, ["model.bin"], MODELS_DIR / "glotlid")
        manifest["models"].append(entry)
        print("Next: compress it with scripts/compress_glotlid.py (see its docstring).")

    (MODELS_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {(MODELS_DIR / 'MANIFEST.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
