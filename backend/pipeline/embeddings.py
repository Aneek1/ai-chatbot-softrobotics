from pathlib import Path
from typing import Protocol

E5_MODEL = "intfloat/multilingual-e5-small"
_registered = False


class Embedder(Protocol):
    dim: int

    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...


def _register_e5() -> None:
    global _registered
    if _registered:
        return
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    TextEmbedding.add_custom_model(
        model=E5_MODEL,
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf=E5_MODEL),
        dim=384,
        model_file="onnx/model.onnx",
    )
    _registered = True


class E5Embedder:
    dim = 384

    def __init__(self, model_dir: Path):
        if not (model_dir / "onnx" / "model.onnx").exists():
            raise FileNotFoundError(
                f"Embedding model not found in {model_dir}. Run scripts/download_models.py first."
            )
        from fastembed import TextEmbedding

        _register_e5()
        # Loading from a plain local folder avoids the Hugging Face cache, which needs symlinks
        # and long paths on Windows.
        self._model = TextEmbedding(E5_MODEL, specific_model_path=str(model_dir))

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed([f"query: {t}" for t in texts])]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed([f"passage: {t}" for t in texts])]
