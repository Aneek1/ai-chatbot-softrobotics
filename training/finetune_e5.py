"""e5-small fine-tune (spec section 16.4, model 3), capped to finish in about 3 hours on an 8 GB GPU.

Usage:
    PYTHONUTF8=1 uv run --group train python -m training.finetune_e5

The run first times a few real steps with the learning rate at zero, then sets the step count so
training plus per-epoch evaluation fits max_minutes. A wall-clock check stops it regardless.
"""

import argparse
import math
import random
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from training.data.rows import Row, read_jsonl
from training.embed_cache import DATA_DIR, load_e5, mean_pool, tokenize
from training.env_report import require_cuda, torch_info, wait_for_gpu
from training.labels import GROUPS, LABELS
from training.metrics import report
from training.runlog import RUNS_DIR, RunLog, new_run_id


@dataclass(frozen=True)
class FinetuneConfig:
    lr: float = 3e-5
    batch_size: int = 32
    max_length: int = 128
    max_epochs: int = 2
    max_minutes: float = 170.0
    warmup_fraction: float = 0.06
    weight_decay: float = 0.01
    dropout: float = 0.1
    per_label_limit: int = 30_000
    patience: int = 1
    eval_batch_size: int = 256
    probe_steps: int = 20
    seed: int = 20260915


class E5Classifier(nn.Module):
    def __init__(self, encoder: nn.Module, n_labels: int, dropout: float):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.config.hidden_size, n_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.classifier(self.dropout(mean_pool(hidden, attention_mask)))


def limit_per_label(rows: list[Row], per_label: int, rng: random.Random) -> list[Row]:
    by_label = defaultdict(list)
    for row in rows:
        by_label[row.label].append(row)
    chosen = [
        row
        for label in sorted(by_label)
        for row in rng.sample(by_label[label], min(per_label, len(by_label[label])))
    ]
    rng.shuffle(chosen)
    return chosen


def plan_steps(
    steps_per_epoch: int, max_epochs: int, seconds_per_step: float, budget_seconds: float, eval_seconds: float
) -> int:
    affordable = int((budget_seconds - eval_seconds * max_epochs) / max(seconds_per_step, 1e-9))
    return max(1, min(steps_per_epoch * max_epochs, affordable))


def warmup_linear(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total - step) / max(1, total - warmup))


def _labels(rows: list[Row], device: torch.device) -> torch.Tensor:
    return torch.tensor([LABELS.index(row.label) for row in rows], device=device)


@torch.inference_mode()
def evaluate(
    model, tokenizer, rows: list[Row], config: FinetuneConfig, device: torch.device
) -> tuple[float, list[str]]:
    model.eval()
    loss, predictions = 0.0, []
    for start in range(0, len(rows), config.eval_batch_size):
        batch = rows[start : start + config.eval_batch_size]
        input_ids, attention_mask = tokenize(tokenizer, [r.text for r in batch], config.max_length, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(input_ids, attention_mask).float()
        loss += F.cross_entropy(logits, _labels(batch, device), reduction="sum").item()
        predictions += [LABELS[i] for i in logits.argmax(dim=-1).tolist()]
    return loss / max(len(rows), 1), predictions


def train_finetune(
    config: FinetuneConfig,
    model: E5Classifier,
    tokenizer,
    train_rows: list[Row],
    val_rows: list[Row],
    runs_dir: Path,
    run_id: str,
    device: str = "cpu",
    env: dict | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    device = torch.device(device)
    model.to(device)
    run = RunLog(
        run_id,
        {"family": "e5_finetune", **asdict(config), "train_rows": len(train_rows), "val_rows": len(val_rows)},
        runs_dir,
        env,
        clock,
    )
    budget = config.max_minutes * 60
    order = list(train_rows)
    rng.shuffle(order)

    def step(optimizer, batch: list[Row]) -> float:
        model.train()
        input_ids, attention_mask = tokenize(tokenizer, [r.text for r in batch], config.max_length, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = F.cross_entropy(model(input_ids, attention_mask).float(), _labels(batch, device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss.item()

    def new_optimizer(lr: float):
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.weight_decay)

    # Probe: real steps at learning rate zero (weights unchanged), then a fresh optimizer.
    probe_optimizer = new_optimizer(0.0)
    probe_batches = [
        order[i * config.batch_size : (i + 1) * config.batch_size] for i in range(config.probe_steps)
    ]
    probe_batches = [b for b in probe_batches if b]
    probe_started = clock()
    for batch in probe_batches:
        step(probe_optimizer, batch)
    seconds_per_step = (clock() - probe_started) / max(len(probe_batches), 1)
    del probe_optimizer
    eval_started = clock()
    evaluate(model, tokenizer, val_rows[: config.eval_batch_size], config, device)
    eval_seconds = (clock() - eval_started) * math.ceil(len(val_rows) / config.eval_batch_size)

    steps_per_epoch = math.ceil(len(order) / config.batch_size)
    planned = plan_steps(
        steps_per_epoch, config.max_epochs, seconds_per_step, budget - run.elapsed(), eval_seconds
    )
    optimizer = new_optimizer(config.lr)
    warmup = max(1, int(planned * config.warmup_fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: warmup_linear(s, planned, warmup))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model_path = run.dir / "model.pt"
    val_gold = [row.label for row in val_rows]
    steps, best_score, best_epoch, best_val, stale, stop_reason, epoch = 0, -1.0, 0, None, 0, "max_epochs", 0
    for epoch in range(1, config.max_epochs + 1):
        rng.shuffle(order)
        total, seen = 0.0, 0
        for start in range(0, len(order), config.batch_size):
            if steps >= planned:
                stop_reason = "step_cap"
                break
            if run.elapsed() >= budget:
                stop_reason = "time_budget"
                break
            batch = order[start : start + config.batch_size]
            total += step(optimizer, batch) * len(batch)
            seen += len(batch)
            scheduler.step()
            steps += 1
        val_loss, predictions = evaluate(model, tokenizer, val_rows, config, device)
        val_report = report(val_gold, predictions, LABELS, GROUPS)
        run.log_epoch(
            epoch,
            train_loss=round(total / seen, 4) if seen else None,
            val_loss=round(val_loss, 4),
            val_macro_f1=val_report["macro_f1"],
            val_group_macro_f1=val_report["group_macro_f1"],
            val_group_f1=val_report["group_f1"],
            steps=steps,
        )
        if val_report["group_macro_f1"] > best_score:
            best_score, best_epoch, best_val, stale = val_report["group_macro_f1"], epoch, val_report, 0
            torch.save(model.state_dict(), model_path)
        else:
            stale += 1
        if stop_reason != "max_epochs":
            break
        if stale >= config.patience:
            stop_reason = "early_stopping"
            break
    peak = round(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else None
    return run.finish(
        family="e5_finetune",
        status="ok",
        best_epoch=best_epoch,
        epochs_run=epoch,
        steps=steps,
        planned_steps=planned,
        steps_per_epoch=steps_per_epoch,
        seconds_per_step=round(seconds_per_step, 4),
        stop_reason=stop_reason,
        peak_gpu_memory_mib=peak,
        val=best_val,
        model_file=model_path.name,
        model_bytes=model_path.stat().st_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--runs", type=Path, default=RUNS_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-minutes", type=float, default=FinetuneConfig.max_minutes)
    args = parser.parse_args()
    config = FinetuneConfig(max_minutes=args.max_minutes)
    if args.device == "cuda":
        wait_for_gpu()
        require_cuda(torch_info())
    encoder, tokenizer = load_e5()
    rng = random.Random(config.seed)
    train_rows = limit_per_label(list(read_jsonl(args.data / "train.jsonl")), config.per_label_limit, rng)
    val_rows = list(read_jsonl(args.data / "val_select.jsonl"))
    model = E5Classifier(encoder, len(LABELS), config.dropout)
    final = train_finetune(
        config, model, tokenizer, train_rows, val_rows, args.runs, new_run_id("e5-finetune"), args.device
    )
    print(
        f"{final['run_id']}: group macro-F1 {final['val']['group_macro_f1']}, {final['steps']} steps, "
        f"stopped by {final['stop_reason']} after {final['wall_clock_seconds'] / 60:.0f} minutes"
    )


if __name__ == "__main__":
    main()
