from pathlib import Path

import pytest

from backend.pipeline.langid import TwoStageDetector
from backend.pipeline.specialist import OnnxClassifierScorer, OnnxHeadScorer, load_specialist
from tests.fakes import FakeEmbedder, FakeScorer

FIXTURES = Path(__file__).parent / "fixtures" / "specialist"


def test_head_scorer_classifies_embedder_output():
    scorer = load_specialist(FIXTURES, "head.onnx", embedder=FakeEmbedder())
    assert isinstance(scorer, OnnxHeadScorer)
    top = scorer.top("3D printing of silicone printing", k=3)
    assert [c.code for c in top] == ["zsm_Latn", "ind_Latn", "other"]
    assert top[0].probability > top[1].probability > top[2].probability
    assert sum(c.probability for c in top) == pytest.approx(1.0, abs=1e-5)


def test_head_scorer_needs_an_embedder():
    with pytest.raises(ValueError, match="needs the e5 embedder"):
        load_specialist(FIXTURES, "head.onnx")


def test_classifier_scorer_pads_and_masks_a_batch():
    scorer = load_specialist(FIXTURES, "classifier.onnx")
    assert isinstance(scorer, OnnxClassifierScorer)
    results = scorer.top_many(["silikon", "lembut lembut lembut lembut"], k=1)
    assert [r[0].code for r in results] == ["ind_Latn", "zsm_Latn"]
    assert scorer.top("silikon", k=1)[0].code == "ind_Latn"


def test_classifier_scorer_truncates_to_max_length():
    scorer = load_specialist(FIXTURES, "classifier.onnx")
    # max_length is 8: "query : " plus six words, so the trailing silikon words are cut off.
    assert scorer.top("lembut " * 6 + "silikon " * 20, k=1)[0].code == "zsm_Latn"


def test_missing_file_names_the_export_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="training/export.py"):
        load_specialist(tmp_path, "lid-specialist-fasttext.ftz")


def test_unsupported_file_type(tmp_path):
    (tmp_path / "model.pkl").write_bytes(b"")
    with pytest.raises(ValueError, match="Unsupported specialist file type: model.pkl"):
        load_specialist(tmp_path, "model.pkl")


def test_unknown_kind_in_sidecar(tmp_path):
    (tmp_path / "x.onnx").write_bytes((FIXTURES / "head.onnx").read_bytes())
    (tmp_path / "x.json").write_text('{"kind": "svm", "labels": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown specialist kind 'svm' in x.json"):
        load_specialist(tmp_path, "x.onnx")


def test_detector_uses_the_onnx_specialist_inside_a_group():
    detector = TwoStageDetector(
        general=FakeScorer([("ind_Latn", 0.7), ("zsm_Latn", 0.3)]),
        min_confidence=0.6,
        specialist=load_specialist(FIXTURES, "classifier.onnx"),
    )
    result = detector.detect("lembut lembut lembut")
    assert result.chosen == "zsm_Latn"
    assert result.stage == "specialist"
