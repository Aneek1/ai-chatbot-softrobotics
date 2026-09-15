"""Detectors compared in the language-ID evaluation (spec section 11.1), mapped onto the specialist's labels.

GlotLID has no Simplified/Traditional labels (it says cmn_Hani), so its Han answers go through the app's
OpenCC rule, as in the app. lid.176 uses ISO 639-1 codes; "hi" and "ur" are split by script.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from backend.pipeline.han_script import han_variant
from backend.pipeline.langid import FastTextScorer, LabelScorer, TwoStageDetector
from backend.pipeline.normalize import normalize, script_profile
from training.data.download import fetch
from training.labels import LABELS, OTHER

UNDETERMINED = "und"
BATCH = 256
LID176_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
LID176_SHA256 = "8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83"
LID176_SOURCE = {
    "name": "fastText lid.176.ftz",
    "url": LID176_URL,
    "sha256": LID176_SHA256,
    "bytes": 938_013,
    "licence": "CC-BY-SA-3.0",
}
ISO_639_1 = {
    "en": "eng_Latn",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ta": "tam_Taml",
}
SPECIALIST_FILES = {
    "specialist-fasttext": "lid-specialist-fasttext.ftz",
    "specialist-e5-head": "lid-specialist-e5-head.onnx",
    "specialist-e5-finetune": "lid-specialist-e5-finetune.onnx",
}
E5_ONNX = Path("multilingual-e5-small") / "onnx" / "model.onnx"

Prediction = tuple[str, float]


def to_label(code: str, text: str) -> str:
    if code in LABELS:
        return code
    if code in ("cmn_Hani", "zh"):
        variant = han_variant(text)
        return UNDETERMINED if variant == "ambiguous" else variant
    if code in ISO_639_1:
        return ISO_639_1[code]
    if code in ("hi", "ur"):
        latin = script_profile(text).dominant == "Latn"
        if code == "hi":
            return "hin_Latn" if latin else "hin_Deva"
        return "urd_Latn" if latin else OTHER
    if code == UNDETERMINED:
        return UNDETERMINED
    return OTHER


class Detector(Protocol):
    name: str

    def predict_many(self, texts: Sequence[str]) -> list[Prediction]: ...


class ScorerDetector:
    def __init__(self, name: str, scorer: LabelScorer):
        self.name = name
        self._scorer = scorer

    def predict_many(self, texts: Sequence[str]) -> list[Prediction]:
        clean = [normalize(text) for text in texts]
        batched = getattr(self._scorer, "top_many", None)
        results: list[Prediction] = []
        for start in range(0, len(clean), BATCH):
            chunk = clean[start : start + BATCH]
            tops = batched(chunk, 1) if batched else [self._scorer.top(text, 1) for text in chunk]
            results += [
                (to_label(top[0].code, text), top[0].probability) if top else (UNDETERMINED, 0.0)
                for top, text in zip(tops, chunk, strict=True)
            ]
        return results


class TwoStageAdapter:
    def __init__(self, name: str, detector: TwoStageDetector):
        self.name = name
        self._detector = detector

    def predict_many(self, texts: Sequence[str]) -> list[Prediction]:
        results = []
        for text in texts:
            detection = self._detector.detect(text)
            probability = detection.candidates[0].probability if detection.candidates else 0.0
            results.append((to_label(detection.chosen, normalize(text)), probability))
        return results


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    files: tuple[Path, ...]
    build: Callable[[], Detector]


def _load_specialist(models_dir: Path, file_name: str) -> LabelScorer:
    from backend.pipeline.embeddings import E5Embedder
    from backend.pipeline.specialist import load_specialist

    embedder = (
        E5Embedder(models_dir / "multilingual-e5-small") if file_name.endswith("e5-head.onnx") else None
    )
    return load_specialist(models_dir, file_name, embedder)


def detector_specs(
    models_dir: Path, include_lid176: bool = True, min_confidence: float = 0.6
) -> list[DetectorSpec]:
    """Detectors whose files exist (lid.176 is downloaded on first use). Nothing is loaded until build()."""
    full, compressed = models_dir / "glotlid" / "model.bin", models_dir / "glotlid-q.ftz"
    specs = []
    if full.exists():
        specs.append(
            DetectorSpec(
                "glotlid-full", (full,), lambda: ScorerDetector("glotlid-full", FastTextScorer(full))
            )
        )
    if compressed.exists():
        specs.append(
            DetectorSpec(
                "glotlid-compressed",
                (compressed,),
                lambda: ScorerDetector("glotlid-compressed", FastTextScorer(compressed)),
            )
        )
    if include_lid176:
        lid176 = models_dir / "lid.176.ftz"
        specs.append(
            DetectorSpec(
                "lid.176",
                (lid176,),
                lambda: ScorerDetector("lid.176", FastTextScorer(fetch(LID176_URL, lid176, LID176_SHA256))),
            )
        )
    for name, file_name in SPECIALIST_FILES.items():
        path = models_dir / file_name
        if not path.exists():
            continue
        extra = (models_dir / E5_ONNX,) if file_name.endswith("e5-head.onnx") else ()
        specs.append(
            DetectorSpec(
                name,
                (path, *extra),
                lambda n=name, f=file_name: ScorerDetector(n, _load_specialist(models_dir, f)),
            )
        )
        if compressed.exists():
            two_stage = f"two-stage+{name}"
            specs.append(
                DetectorSpec(
                    two_stage,
                    (compressed, path, *extra),
                    lambda n=two_stage, f=file_name: TwoStageAdapter(
                        n,
                        TwoStageDetector(
                            FastTextScorer(compressed),
                            min_confidence,
                            specialist=_load_specialist(models_dir, f),
                        ),
                    ),
                )
            )
    return specs
