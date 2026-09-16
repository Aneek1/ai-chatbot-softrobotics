"""Load a trained stage-2 specialist (spec section 16.6) with fasttext-predict or onnxruntime; no PyTorch."""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from backend.pipeline.embeddings import Embedder
from backend.pipeline.langid import Candidate, FastTextScorer, LabelScorer

PREFIX = "query: "
BATCH_SIZE = 64


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _top(probabilities: np.ndarray, labels: Sequence[str], k: int) -> list[Candidate]:
    order = np.argsort(-probabilities, kind="stable")[:k]
    return [Candidate(labels[i], float(probabilities[i])) for i in order]


def _session(path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


class OnnxHeadScorer:
    """Classifier on e5 query embeddings; the app's embedder already adds the "query: " prefix."""

    def __init__(self, model_path: Path, labels: Sequence[str], embedder: Embedder):
        self._session = _session(model_path)
        self._labels = list(labels)
        self._embedder = embedder

    def top(self, text: str, k: int) -> list[Candidate]:
        return self.top_many([text], k)[0]

    def top_many(self, texts: Sequence[str], k: int) -> list[list[Candidate]]:
        results = []
        for start in range(0, len(texts), BATCH_SIZE):
            vectors = np.asarray(
                self._embedder.embed_queries(list(texts[start : start + BATCH_SIZE])), np.float32
            )
            logits = self._session.run(["logits"], {"embedding": vectors})[0]
            results += [_top(row, self._labels, k) for row in _softmax(logits)]
        return results


class OnnxClassifierScorer:
    """Fine-tuned e5 encoder plus classification head, fed token ids from the e5 tokenizer."""

    def __init__(
        self, model_path: Path, labels: Sequence[str], tokenizer_path: Path, max_length: int, pad_id: int
    ):
        from tokenizers import Tokenizer

        self._session = _session(model_path)
        self._labels = list(labels)
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length)
        self._pad_id = pad_id

    def top(self, text: str, k: int) -> list[Candidate]:
        return self.top_many([text], k)[0]

    def top_many(self, texts: Sequence[str], k: int) -> list[list[Candidate]]:
        results = []
        for start in range(0, len(texts), BATCH_SIZE):
            encodings = self._tokenizer.encode_batch([PREFIX + t for t in texts[start : start + BATCH_SIZE]])
            width = max(len(e.ids) for e in encodings)
            input_ids = np.full((len(encodings), width), self._pad_id, dtype=np.int64)
            attention_mask = np.zeros((len(encodings), width), dtype=np.int64)
            for row, encoding in enumerate(encodings):
                input_ids[row, : len(encoding.ids)] = encoding.ids
                attention_mask[row, : len(encoding.ids)] = 1
            logits = self._session.run(
                ["logits"], {"input_ids": input_ids, "attention_mask": attention_mask}
            )[0]
            results += [_top(row, self._labels, k) for row in _softmax(logits)]
        return results


def load_specialist(models_dir: Path, name: str, embedder: Embedder | None = None) -> LabelScorer:
    path = models_dir / name
    if not path.exists():
        raise FileNotFoundError(
            f"Specialist model not found at {path}. "
            "Export one with training/export.py (see training/README.md)."
        )
    if path.suffix in (".ftz", ".bin"):
        return FastTextScorer(path)
    if path.suffix != ".onnx":
        raise ValueError(f"Unsupported specialist file type: {path.name}")
    sidecar = path.with_suffix(".json")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    if meta["kind"] == "e5_head":
        if embedder is None:
            raise ValueError(f"{path.name} classifies e5 embeddings, so it needs the e5 embedder")
        return OnnxHeadScorer(path, meta["labels"], embedder)
    if meta["kind"] == "e5_finetune":
        return OnnxClassifierScorer(
            path, meta["labels"], models_dir / meta["tokenizer"], meta["max_length"], meta["pad_id"]
        )
    raise ValueError(f"Unknown specialist kind {meta['kind']!r} in {sidecar.name}")
