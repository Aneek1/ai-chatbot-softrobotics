import json
from pathlib import Path

import pytest

from training.data.rows import Row, write_jsonl
from training.export import OUTPUT_NAMES, export_fasttext
from training.labels import LABELS
from training.runlog import RunLog


class FakeQuantizable:
    def __init__(self):
        self.quantize_kwargs = None

    def quantize(self, **kwargs):
        self.quantize_kwargs = kwargs
        self.lines = Path(kwargs["input"]).read_text(encoding="utf-8").splitlines()

    def save_model(self, path):
        Path(path).write_bytes(b"ftz")


class FakeFastText:
    def __init__(self):
        self.model = FakeQuantizable()

    def load_model(self, path):
        assert path.endswith("model.bin")
        return self.model


def test_output_names():
    assert OUTPUT_NAMES["fasttext"].endswith(".ftz")
    assert OUTPUT_NAMES["e5_head"].endswith(".onnx") and OUTPUT_NAMES["e5_finetune"].endswith(".onnx")


def test_fasttext_export_quantizes_with_the_runs_training_rows(tmp_path):
    data = tmp_path / "data"
    write_jsonl(
        [Row("soft robot", "eng_Latn", "openlid", "a"), Row("robot", "eng_Latn", "openlid", "a", "crop")],
        data / "train.jsonl",
    )
    config = {
        "epoch": 5,
        "lr": 0.5,
        "dim": 16,
        "minn": 1,
        "maxn": 4,
        "word_ngrams": 1,
        "bucket": 10,
        "loss": "softmax",
        "include_synthetic": False,
        "seed": 1,
    }
    run = RunLog("fasttext-02", config, runs_dir=tmp_path / "runs", env={})
    fake = FakeFastText()
    out = export_fasttext(
        run.dir, tmp_path / "models" / "specialist.ftz", data_dir=data, fasttext_module=fake
    )
    assert out.read_bytes() == b"ftz"
    assert fake.model.lines == ["__label__eng_Latn soft robot"]
    assert fake.model.quantize_kwargs["retrain"] is True and fake.model.quantize_kwargs["qnorm"] is True
    assert not (run.dir / "quantize-train.txt").exists()
    sidecar = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert (
        sidecar["kind"] == "fasttext"
        and sidecar["run_id"] == "fasttext-02"
        and sidecar["labels"] == list(LABELS)
    )


def test_head_export_matches_pytorch(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxscript")
    from training.embed_head import MLPHead
    from training.export import export_head

    config = {
        "hidden": 8,
        "lr": 0.001,
        "dropout": 0.1,
        "max_epochs": 1,
        "patience": 1,
        "batch_size": 4,
        "weight_decay": 0.0,
        "seed": 1,
        "embedding_dim": 6,
    }
    run = RunLog("e5-head-00", config, runs_dir=tmp_path, env={})
    torch.manual_seed(0)
    torch.save(MLPHead(6, 8, len(LABELS), 0.1).state_dict(), run.dir / "model.pt")
    out = export_head(run.dir, tmp_path / "head.onnx")
    sidecar = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["kind"] == "e5_head" and sidecar["prefix"] == "query: "
    assert sidecar["onnx_max_abs_diff"] < 1e-4


def test_finetune_export_matches_pytorch_with_dynamic_lengths(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxscript")
    from tests.lid_torch_fixtures import tiny_encoder, tiny_tokenizer
    from training.export import export_finetune
    from training.finetune_e5 import E5Classifier

    run = RunLog("e5-finetune", {"dropout": 0.1, "max_length": 16}, runs_dir=tmp_path, env={})
    torch.save(E5Classifier(tiny_encoder(), len(LABELS), 0.1).state_dict(), run.dir / "model.pt")
    tokenizer = tiny_tokenizer()
    out = export_finetune(
        run.dir,
        tmp_path / "tune.onnx",
        encoder=tiny_encoder(seed=5),
        tokenizer=tokenizer,
        tokenizer_name="tokenizer.json",
    )
    sidecar = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["kind"] == "e5_finetune"
    assert sidecar["max_length"] == 16 and sidecar["pad_id"] == tokenizer.pad_token_id
    assert sidecar["onnx_max_abs_diff"] < 1e-4
