import random

import pytest

torch = pytest.importorskip("torch")

from tests.lid_torch_fixtures import tiny_encoder, tiny_tokenizer  # noqa: E402
from training.data.rows import Row  # noqa: E402
from training.finetune_e5 import (  # noqa: E402
    E5Classifier,
    FinetuneConfig,
    limit_per_label,
    plan_steps,
    train_finetune,
    warmup_linear,
)
from training.runlog import read_metrics  # noqa: E402

TRAIN = [
    Row("soft robot air", "eng_Latn", "openlid", "a"),
    Row("robot lunak udara", "ind_Latn", "openlid", "a"),
    Row("robot lembut udara", "zsm_Latn", "openlid", "a"),
] * 8
VAL = TRAIN[:3]


class StepClock:
    def __init__(self, step: float):
        self.now, self.step = 0.0, step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def test_plan_steps_fits_the_budget():
    assert (
        plan_steps(
            steps_per_epoch=1000, max_epochs=2, seconds_per_step=0.5, budget_seconds=600, eval_seconds=50
        )
        == 1000
    )
    assert (
        plan_steps(
            steps_per_epoch=100, max_epochs=2, seconds_per_step=0.5, budget_seconds=600, eval_seconds=50
        )
        == 200
    )
    assert (
        plan_steps(steps_per_epoch=100, max_epochs=2, seconds_per_step=10, budget_seconds=5, eval_seconds=50)
        == 1
    )


def test_warmup_then_linear_decay():
    assert warmup_linear(0, total=100, warmup=10) == pytest.approx(0.1)
    assert warmup_linear(9, total=100, warmup=10) == pytest.approx(1.0)
    assert warmup_linear(55, total=100, warmup=10) == pytest.approx(0.5)
    assert warmup_linear(100, total=100, warmup=10) == 0.0


def test_limit_per_label_is_seeded_and_capped():
    rows = [Row(f"t{i}", "eng_Latn", "openlid", "a") for i in range(10)] + [
        Row("x", "ind_Latn", "openlid", "a")
    ]
    first = limit_per_label(rows, 4, random.Random(1))
    assert len(first) == 5
    assert first == limit_per_label(rows, 4, random.Random(1))


def test_default_config_is_capped():
    config = FinetuneConfig()
    assert config.max_epochs <= 4 and config.max_length == 128 and config.max_minutes <= 180


def test_short_run_on_cpu_writes_records(tmp_path):
    model = E5Classifier(tiny_encoder(), 12, dropout=0.0)
    config = FinetuneConfig(lr=1e-3, batch_size=8, max_epochs=2, probe_steps=2, eval_batch_size=4, patience=5)
    final = train_finetune(config, model, tiny_tokenizer(), TRAIN, VAL, tmp_path, "ft-e5", env={})
    assert final["stop_reason"] == "max_epochs"
    assert final["steps"] == 6 and final["planned_steps"] == 6
    assert final["val"]["count"] == 3
    assert final["peak_gpu_memory_mib"] is None
    assert (tmp_path / "ft-e5" / "model.pt").exists()
    assert [m["steps"] for m in read_metrics(tmp_path / "ft-e5")] == [3, 6]


def test_probe_leaves_weights_unchanged(tmp_path):
    model = E5Classifier(tiny_encoder(), 12, dropout=0.0)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    config = FinetuneConfig(batch_size=8, max_epochs=1, probe_steps=3, eval_batch_size=4, max_minutes=0.0)
    final = train_finetune(config, model, tiny_tokenizer(), TRAIN, VAL, tmp_path, "probe", env={})
    assert final["steps"] == 0
    assert final["stop_reason"] == "time_budget"
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_wall_clock_budget_stops_mid_epoch(tmp_path):
    model = E5Classifier(tiny_encoder(), 12, dropout=0.0)
    config = FinetuneConfig(batch_size=2, max_epochs=3, probe_steps=1, eval_batch_size=4, max_minutes=1.0)
    final = train_finetune(
        config, model, tiny_tokenizer(), TRAIN, VAL, tmp_path, "budget", env={}, clock=StepClock(4.0)
    )
    assert final["stop_reason"] in {"time_budget", "step_cap"}
    assert final["steps"] < 12 * 3
    assert final["epochs_run"] == 1
