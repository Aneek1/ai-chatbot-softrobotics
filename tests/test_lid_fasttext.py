from pathlib import Path

from training.data.rows import Row, write_jsonl
from training.fasttext_train import (
    SWEEP,
    FastTextConfig,
    config_of_run,
    to_line,
    train_one,
    write_training_file,
)
from training.runlog import read_json, read_metrics

ROWS = [
    Row("soft robot", "eng_Latn", "openlid", "lti"),
    Row("robot lunak", "ind_Latn", "openlid", "lti"),
    Row("robot", "eng_Latn", "openlid", "lti", "crop"),
]


class FakeModel:
    def __init__(self, answers: dict[str, str]):
        self._answers = answers
        self.saved: list[str] = []

    def predict(self, text: str, k: int):
        best = self._answers[text]
        others = [f"__label__{label}" for label in ("eng_Latn", "ind_Latn") if label != best]
        return (f"__label__{best}", *others), (0.8, *[0.2 / len(others)] * len(others))

    def save_model(self, path: str) -> None:
        Path(path).write_bytes(b"model")
        self.saved.append(path)


class FakeFastText:
    def __init__(self, model=None, error: Exception | None = None):
        self.model, self.error, self.kwargs = model, error, None

    def train_supervised(self, **kwargs):
        self.kwargs = kwargs
        self.lines = Path(kwargs["input"]).read_text(encoding="utf-8").splitlines()
        if self.error:
            raise self.error
        return self.model


def make_data(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    write_jsonl(ROWS, data / "train.jsonl")
    write_jsonl(ROWS[:2], data / "val_select.jsonl")
    return data


def test_line_format():
    assert to_line(ROWS[0]) == "__label__eng_Latn soft robot"


def test_training_file_can_leave_out_synthetic_rows(tmp_path):
    assert write_training_file(ROWS, tmp_path / "all.txt", include_synthetic=True) == 3
    assert write_training_file(ROWS, tmp_path / "natural.txt", include_synthetic=False) == 2


def test_sweep_is_small_and_covers_every_listed_value():
    assert len(SWEEP) == 6
    assert {c.epoch for c in SWEEP} == {5, 10, 25}
    assert {c.lr for c in SWEEP} == {0.1, 0.5, 1.0}
    assert {c.dim for c in SWEEP} == {16, 64}
    assert {(c.minn, c.maxn) for c in SWEEP} == {(1, 4), (2, 5)}


def test_train_one_records_the_run(tmp_path):
    data = make_data(tmp_path)
    fake = FakeFastText(FakeModel({"soft robot": "eng_Latn", "robot lunak": "eng_Latn"}))
    config = FastTextConfig(epoch=5, lr=0.5, dim=16, minn=1, maxn=4)
    final = train_one(config, data, tmp_path / "runs", "ft-00", fasttext_module=fake, env={})

    assert fake.kwargs["minn"] == 1 and fake.kwargs["wordNgrams"] == 1 and fake.kwargs["seed"] == 20260915
    assert fake.lines == [
        "__label__eng_Latn soft robot",
        "__label__ind_Latn robot lunak",
        "__label__eng_Latn robot",
    ]
    run_dir = tmp_path / "runs" / "ft-00"
    assert not (run_dir / "train.txt").exists()
    assert (run_dir / "model.bin").exists()
    assert final["status"] == "ok"
    assert final["val"]["per_label"]["ind_Latn"]["recall"] == 0.0
    assert final["val"]["confusions"]["malay_indonesian"]["ind_Latn"]["(outside group)"] == 1
    metrics = read_metrics(run_dir)
    assert len(metrics) == 1 and metrics[0]["train_loss"] is None and metrics[0]["val_loss"] > 0
    stored = read_json(run_dir / "config.json")["config"]
    assert stored["family"] == "fasttext"
    # every thread by default, which fastText cannot reproduce run to run, seed or no seed
    assert fake.kwargs["thread"] == stored["threads"] >= 1
    assert stored["reproducible"] is (stored["threads"] == 1)
    assert config_of_run(run_dir) == config


def test_diverged_training_is_recorded_not_raised(tmp_path):
    data = make_data(tmp_path)
    fake = FakeFastText(error=RuntimeError("Encountered NaN."))
    final = train_one(
        FastTextConfig(epoch=10, lr=1.0, dim=64, minn=1, maxn=4),
        data,
        tmp_path / "runs",
        "ft-03",
        fasttext_module=fake,
        env={},
        threads=1,
    )
    assert fake.kwargs["thread"] == 1
    stored = read_json(tmp_path / "runs" / "ft-03" / "config.json")["config"]
    assert stored["threads"] == 1 and stored["reproducible"] is True
    assert final["status"] == "failed"
    assert final["error"] == "Encountered NaN."
    assert not (tmp_path / "runs" / "ft-03" / "train.txt").exists()
