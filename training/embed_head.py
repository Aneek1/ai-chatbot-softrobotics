"""Classifier on frozen e5 embeddings (spec section 16.4, model 2).

Usage (after training.embed_cache):
    PYTHONUTF8=1 uv run --group train python -m training.embed_head --sweep
"""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from training.embed_cache import DATA_DIR, load_split
from training.env_report import require_cuda, torch_info, wait_for_gpu
from training.labels import GROUPS, LABELS
from training.metrics import report
from training.runlog import RUNS_DIR, RunLog
from training.sweep import run_grid


@dataclass(frozen=True)
class HeadConfig:
    hidden: int
    lr: float
    dropout: float
    max_epochs: int = 30
    patience: int = 3
    batch_size: int = 1024
    weight_decay: float = 0.01
    seed: int = 20260915


SWEEP = tuple(
    HeadConfig(hidden=hidden, lr=lr, dropout=dropout)
    for hidden in (128, 512)
    for lr in (1e-3, 3e-4)
    for dropout in (0.1, 0.3)
)


class MLPHead(nn.Module):
    def __init__(self, dim: int, hidden: int, n_labels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_labels)
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


@torch.inference_mode()
def evaluate(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 8192
) -> tuple[float, list[str]]:
    model.eval()
    loss, predictions = 0.0, []
    for start in range(0, len(x), batch_size):
        logits = model(x[start : start + batch_size].float())
        loss += F.cross_entropy(logits, y[start : start + batch_size], reduction="sum").item()
        predictions += [LABELS[i] for i in logits.argmax(dim=-1).tolist()]
    return loss / max(len(x), 1), predictions


def train_head(
    config: HeadConfig,
    train: tuple[np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray],
    runs_dir: Path,
    run_id: str,
    device: str = "cpu",
    env: dict | None = None,
) -> dict:
    torch.manual_seed(config.seed)
    device = torch.device(device)
    train_x, train_y = torch.as_tensor(train[0]).to(device), torch.as_tensor(train[1]).to(device)
    val_x, val_y = torch.as_tensor(val[0]).to(device), torch.as_tensor(val[1]).to(device)
    val_gold = [LABELS[i] for i in val[1].tolist()]
    model = MLPHead(train_x.shape[1], config.hidden, len(LABELS), config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    run = RunLog(
        run_id,
        {
            "family": "e5_head",
            **asdict(config),
            "embedding_dim": int(train_x.shape[1]),
            "train_rows": len(train_x),
            "val_rows": len(val_x),
        },
        runs_dir,
        env,
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    model_path = run.dir / "model.pt"
    best_score, best_epoch, best_val, stale, epoch = -1.0, 0, None, 0, 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total = 0.0
        permutation = torch.randperm(len(train_x), generator=generator).to(device)
        for start in range(0, len(permutation), config.batch_size):
            index = permutation[start : start + config.batch_size]
            loss = F.cross_entropy(model(train_x[index].float()), train_y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(index)
        val_loss, predictions = evaluate(model, val_x, val_y)
        val_report = report(val_gold, predictions, LABELS, GROUPS)
        run.log_epoch(
            epoch,
            train_loss=round(total / len(train_x), 4),
            val_loss=round(val_loss, 4),
            val_macro_f1=val_report["macro_f1"],
            val_group_macro_f1=val_report["group_macro_f1"],
            val_group_f1=val_report["group_f1"],
        )
        if val_report["group_macro_f1"] > best_score:
            best_score, best_epoch, best_val, stale = val_report["group_macro_f1"], epoch, val_report, 0
            torch.save(model.state_dict(), model_path)
        else:
            stale += 1
            if stale >= config.patience:
                break
    return run.finish(
        family="e5_head",
        status="ok",
        best_epoch=best_epoch,
        epochs_run=epoch,
        val=best_val,
        model_file=model_path.name,
        model_bytes=model_path.stat().st_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_DIR / "emb")
    parser.add_argument("--runs", type=Path, default=RUNS_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda":
        wait_for_gpu()
        require_cuda(torch_info())
    train_x, train_y, _ = load_split(args.data, "train")
    val_x, val_y, _ = load_split(args.data, "val_select")
    configs = SWEEP if args.sweep else SWEEP[:1]
    path = run_grid(
        "e5-head",
        [asdict(c) for c in configs],
        lambda cfg, run_id: train_head(
            HeadConfig(**cfg), (train_x, train_y), (val_x, val_y), args.runs, run_id, args.device
        ),
        args.runs,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
