"""Write the tiny ONNX specialists used by tests/test_specialist.py.

They are committed, so the tests need only onnxruntime. Regenerating needs the train group (onnx):
    uv run --group train python tests/fixtures/make_specialist_fixtures.py
"""

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from tokenizers import Tokenizer, models, pre_tokenizers

OUT = Path(__file__).parent / "specialist"
LABELS = ["ind_Latn", "zsm_Latn", "other"]
OPSET = [helper.make_opsetid("", 17)]
VOCAB = {"[PAD]": 0, "[UNK]": 1, "query": 2, ":": 3, "silikon": 4, "lembut": 5}


def save(graph: onnx.GraphProto, name: str, sidecar: dict) -> None:
    model = helper.make_model(graph, opset_imports=OPSET, producer_name="make_specialist_fixtures")
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, OUT / f"{name}.onnx")
    (OUT / f"{name}.json").write_text(
        json.dumps({**sidecar, "labels": LABELS}, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def head() -> None:
    # FakeEmbedder vectors are [silicone, printing, market, 0.01] keyword counts.
    weight = np.zeros((4, 3), dtype=np.float32)
    weight[0, 0] = weight[1, 1] = weight[2, 2] = 5.0
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["embedding", "W"], ["scores"]),
            helper.make_node("Add", ["scores", "B"], ["logits"]),
        ],
        "tiny_head",
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", 4])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 3])],
        [numpy_helper.from_array(weight, "W"), numpy_helper.from_array(np.zeros(3, np.float32), "B")],
    )
    save(graph, "head", {"kind": "e5_head"})


def classifier() -> None:
    table = np.zeros((len(VOCAB), 3), dtype=np.float32)
    table[VOCAB["[PAD]"]] = [0, 0, 9]  # padding points at "other", so a missing mask shows up in tests
    table[VOCAB["silikon"]] = [4, 0, 0]
    table[VOCAB["lembut"]] = [0, 4, 0]
    nodes = [
        helper.make_node("Gather", ["E", "input_ids"], ["tokens"]),
        helper.make_node("Cast", ["attention_mask"], ["mask"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["mask", "axis2"], ["mask3"]),
        helper.make_node("Mul", ["tokens", "mask3"], ["masked"]),
        helper.make_node("ReduceSum", ["masked", "axis1"], ["summed"], keepdims=0),
        helper.make_node("ReduceSum", ["mask3", "axis1"], ["count"], keepdims=0),
        helper.make_node("Div", ["summed", "count"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny_classifier",
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "sequence"]),
        ],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 3])],
        [
            numpy_helper.from_array(table, "E"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "axis1"),
            numpy_helper.from_array(np.array([2], dtype=np.int64), "axis2"),
        ],
    )
    tokenizer = Tokenizer(models.WordLevel(VOCAB, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(OUT / "tokenizer.json"))
    save(
        graph,
        "classifier",
        {"kind": "e5_finetune", "max_length": 8, "pad_id": 0, "tokenizer": "tokenizer.json"},
    )


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    head()
    classifier()
    print(f"Wrote fixtures to {OUT}")
