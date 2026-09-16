"""Embed each split once with multilingual-e5-small (spec section 16.4, model 2).

Usage:
    PYTHONUTF8=1 uv run --group train python -m training.embed_cache

Writes data/cache/lid/emb/<split>.x.npy (float16), <split>.y.npy (label index) and
<split>.synthetic.npy (bool). Embeddings use the app's question prefix and mean pooling, so a head
trained on them works on the vectors fastembed produces at runtime.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from training.data.rows import Row, read_jsonl
from training.env_report import require_cuda, torch_info, wait_for_gpu
from training.labels import LABELS

REPO_ROOT = Path(__file__).resolve().parents[1]
E5_DIR = REPO_ROOT / "models" / "multilingual-e5-small"
DATA_DIR = REPO_ROOT / "data" / "cache" / "lid"
PREFIX = "query: "
MAX_LENGTH = 128


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)


def load_e5(model_dir: Path = E5_DIR):
    from transformers import AutoModel, AutoTokenizer

    if not (model_dir / "model.safetensors").exists():
        raise FileNotFoundError(
            f"PyTorch weights not found in {model_dir}. "
            "Run: uv run python scripts/download_models.py --e5-torch"
        )
    return AutoModel.from_pretrained(str(model_dir)), AutoTokenizer.from_pretrained(str(model_dir))


def tokenize(
    tokenizer, texts: list[str], max_length: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = tokenizer(
        [PREFIX + text for text in texts],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return batch["input_ids"].to(device), batch["attention_mask"].to(device)


@torch.inference_mode()
def embed_texts(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 256,
    max_length: int = MAX_LENGTH,
    device: str = "cpu",
) -> np.ndarray:
    device = torch.device(device)
    model = model.to(device).eval()
    # Similar lengths share a batch, so less compute goes into padding.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    vectors = np.zeros((len(texts), model.config.hidden_size), dtype=np.float16)
    for start in range(0, len(order), batch_size):
        index = order[start : start + batch_size]
        input_ids, attention_mask = tokenize(tokenizer, [texts[i] for i in index], max_length, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = torch.nn.functional.normalize(mean_pool(hidden.float(), attention_mask), dim=-1)
        vectors[index] = pooled.cpu().numpy().astype(np.float16)
    return vectors


def cache_split(
    model, tokenizer, rows: list[Row], out_dir: Path, split: str, batch_size: int, device: str
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors = embed_texts(model, tokenizer, [row.text for row in rows], batch_size=batch_size, device=device)
    np.save(out_dir / f"{split}.x.npy", vectors)
    np.save(out_dir / f"{split}.y.npy", np.array([LABELS.index(row.label) for row in rows], dtype=np.int64))
    np.save(
        out_dir / f"{split}.synthetic.npy", np.array([row.synthetic is not None for row in rows], dtype=bool)
    )
    return len(rows)


def load_split(out_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.load(out_dir / f"{split}.{part}.npy") for part in ("x", "y", "synthetic"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--splits", default="train,val_select,test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device == "cuda":
        wait_for_gpu()
        require_cuda(torch_info())
    model, tokenizer = load_e5()
    out_dir = args.data / "emb"
    manifest = {
        "model_dir": str(E5_DIR.relative_to(REPO_ROOT)),
        "prefix": PREFIX,
        "max_length": MAX_LENGTH,
        "labels": list(LABELS),
        "splits": {},
    }
    for split in args.splits.split(","):
        started = time.monotonic()
        count = cache_split(
            model,
            tokenizer,
            list(read_jsonl(args.data / f"{split}.jsonl")),
            out_dir,
            split,
            args.batch_size,
            args.device,
        )
        manifest["splits"][split] = {"rows": count, "seconds": round(time.monotonic() - started, 1)}
        print(f"{split}: {count:,} rows in {manifest['splits'][split]['seconds']} s")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
