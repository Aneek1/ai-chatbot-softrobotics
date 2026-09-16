"""Export chosen models (spec section 16.6) so the app runs them without PyTorch.

    # fastText winner, quantized to .ftz (isolated environment, as for training):
    PYTHONUTF8=1 uv run --no-project --python 3.12 --with fasttext-wheel==0.9.2 --with numpy==1.26.4 \
        python -m training.export fasttext runs/<run_id>
    # neural winners, to ONNX:
    PYTHONUTF8=1 uv run --group train python -m training.export e5_head runs/<run_id>
    PYTHONUTF8=1 uv run --group train python -m training.export e5_finetune runs/<run_id>

Each model gets a JSON sidecar (same name, .json) with its kind, labels and the run it came from.
"""

import argparse
import importlib
import json
import os
from dataclasses import fields
from pathlib import Path

from training.labels import LABELS

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
DATA_DIR = REPO_ROOT / "data" / "cache" / "lid"
OUTPUT_NAMES = {
    "fasttext": "lid-specialist-fasttext.ftz",
    "e5_head": "lid-specialist-e5-head.onnx",
    "e5_finetune": "lid-specialist-e5-finetune.onnx",
}


def _stored_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["config"]


def write_sidecar(model_path: Path, kind: str, run_id: str, **extra) -> Path:
    path = model_path.with_suffix(".json")
    payload = {
        "kind": kind,
        "labels": list(LABELS),
        "run_id": run_id,
        "bytes": model_path.stat().st_size,
        **extra,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def export_fasttext(run_dir: Path, out_path: Path, data_dir: Path = DATA_DIR, fasttext_module=None) -> Path:
    from training.data.rows import read_jsonl
    from training.fasttext_train import config_of_run, write_training_file

    fasttext = fasttext_module or importlib.import_module("fasttext")
    config = config_of_run(run_dir)
    model = fasttext.load_model(str(run_dir / "model.bin"))
    train_file = run_dir / "quantize-train.txt"
    write_training_file(read_jsonl(data_dir / "train.jsonl"), train_file, config.include_synthetic)
    try:
        # retrain=True fine-tunes after pruning to the 100,000 most useful n-gram buckets.
        model.quantize(
            input=str(train_file),
            qnorm=True,
            retrain=True,
            cutoff=100_000,
            dsub=2,
            thread=os.cpu_count() or 1,
        )
    finally:
        train_file.unlink(missing_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_path))
    write_sidecar(out_path, "fasttext", run_dir.name)
    return out_path


def _max_abs_diff(session, feeds: dict, expected) -> float:
    import numpy as np

    return float(np.abs(session.run(["logits"], feeds)[0] - expected).max())


def export_head(run_dir: Path, out_path: Path) -> Path:
    import numpy as np
    import onnxruntime as ort
    import torch

    from training.embed_cache import PREFIX
    from training.embed_head import HeadConfig, MLPHead

    stored = _stored_config(run_dir)
    config = HeadConfig(**{f.name: stored[f.name] for f in fields(HeadConfig)})
    model = MLPHead(stored["embedding_dim"], config.hidden, len(LABELS), config.dropout)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (torch.zeros(2, stored["embedding_dim"]),),
        str(out_path),
        input_names=["embedding"],
        output_names=["logits"],
        dynamic_axes={"embedding": {0: "batch"}, "logits": {0: "batch"}},
        external_data=False,
    )
    sample = np.random.default_rng(0).normal(size=(5, stored["embedding_dim"])).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(sample)).numpy()
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    diff = _max_abs_diff(session, {"embedding": sample}, expected)
    write_sidecar(
        out_path,
        "e5_head",
        run_dir.name,
        prefix=PREFIX,
        embedding_dim=stored["embedding_dim"],
        onnx_max_abs_diff=diff,
    )
    return out_path


def export_finetune(
    run_dir: Path,
    out_path: Path,
    e5_dir: Path = MODELS_DIR / "multilingual-e5-small",
    encoder=None,
    tokenizer=None,
    tokenizer_name: str = "multilingual-e5-small/tokenizer.json",
) -> Path:
    import onnxruntime as ort
    import torch

    from training.embed_cache import PREFIX, tokenize
    from training.finetune_e5 import E5Classifier

    stored = _stored_config(run_dir)
    if encoder is None or tokenizer is None:
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        encoder = AutoModel.from_config(AutoConfig.from_pretrained(str(e5_dir)))  # weights come from model.pt
        tokenizer = AutoTokenizer.from_pretrained(str(e5_dir))
    model = E5Classifier(encoder, len(LABELS), stored["dropout"])
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    model.eval()
    cpu = torch.device("cpu")
    example = tokenize(tokenizer, ["soft robot", "silicone soft robot air"], stored["max_length"], cpu)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        str(out_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        external_data=False,
    )
    check = tokenize(
        tokenizer, ["robot", "a longer line about soft robot air", "air"], stored["max_length"], cpu
    )
    with torch.no_grad():
        expected = model(*check).numpy()
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    diff = _max_abs_diff(
        session, {"input_ids": check[0].numpy(), "attention_mask": check[1].numpy()}, expected
    )
    write_sidecar(
        out_path,
        "e5_finetune",
        run_dir.name,
        prefix=PREFIX,
        max_length=stored["max_length"],
        pad_id=tokenizer.pad_token_id,
        tokenizer=tokenizer_name,
        onnx_max_abs_diff=diff,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=sorted(OUTPUT_NAMES))
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, help="defaults to models/<family file name>")
    args = parser.parse_args()
    out = args.out or MODELS_DIR / OUTPUT_NAMES[args.family]
    exporters = {"fasttext": export_fasttext, "e5_head": export_head, "e5_finetune": export_finetune}
    path = exporters[args.family](args.run_dir, out)
    print(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB) and {path.with_suffix('.json').name}")


if __name__ == "__main__":
    main()
