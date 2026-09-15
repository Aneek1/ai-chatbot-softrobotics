from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from backend.pipeline.han_script import han_variant
from backend.pipeline.languages import group_of
from backend.pipeline.normalize import normalize, script_profile

Stage = Literal["general", "specialist"]


@dataclass(frozen=True)
class Candidate:
    code: str
    probability: float


@dataclass(frozen=True)
class Detection:
    candidates: tuple[Candidate, ...]
    chosen: str
    uncertain: bool
    stage: Stage


class LabelScorer(Protocol):
    def top(self, text: str, k: int) -> list[Candidate]: ...


class LanguageDetector(Protocol):
    def detect(self, text: str) -> Detection: ...


class FastTextScorer:
    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Language-ID model not found at {model_path}. Run scripts/download_models.py "
                "and scripts/compress_glotlid.py first."
            )
        import fasttext

        self._model = fasttext.load_model(str(model_path))

    def top(self, text: str, k: int) -> list[Candidate]:
        labels, probabilities = self._model.predict(text, k=k)
        return [
            Candidate(label.removeprefix("__label__"), float(p))
            for label, p in zip(labels, probabilities, strict=True)
        ]


class TwoStageDetector:
    def __init__(self, general: LabelScorer, min_confidence: float, specialist: LabelScorer | None = None):
        self._general = general
        self._specialist = specialist
        self._min_confidence = min_confidence

    def detect(self, text: str) -> Detection:
        clean = normalize(text)
        if not clean:
            return Detection(candidates=(), chosen="und", uncertain=True, stage="general")

        candidates = self._general.top(clean, k=5)
        top = candidates[0]
        stage: Stage = "general"
        group = group_of(top.code)

        if group is not None and self._specialist is not None:
            candidates = self._specialist.top(clean, k=3)
            top = candidates[0]
            stage = "specialist"
        elif group == "han":
            stage = "specialist"
            variant = han_variant(clean)
            if variant == "ambiguous":
                return Detection(tuple(candidates[:3]), top.code, uncertain=True, stage=stage)
            top = Candidate(variant, top.probability)
            candidates = [top, *candidates[1:]]

        uncertain = top.probability < self._min_confidence or script_profile(clean).mixed
        return Detection(tuple(candidates[:3]), top.code, uncertain=uncertain, stage=stage)
