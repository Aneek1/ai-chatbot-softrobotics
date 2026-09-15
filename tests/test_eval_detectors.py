import shutil
from pathlib import Path

from backend.pipeline.langid import TwoStageDetector
from eval.detectors import UNDETERMINED, ScorerDetector, TwoStageAdapter, detector_specs, to_label
from tests.fakes import FakeScorer

SPECIALIST_FIXTURES = Path(__file__).parent / "fixtures" / "specialist"


def test_specialist_labels_pass_through():
    assert to_label("zsm_Latn", "apa khabar") == "zsm_Latn"
    assert to_label("other", "bonjour") == "other"


def test_han_codes_use_the_opencc_rule():
    assert to_label("cmn_Hani", "如何制作软气动执行器") == "zho_Hans"
    assert to_label("zh", "如何製作軟氣動執行器") == "zho_Hant"
    assert to_label("cmn_Hani", "中文") == UNDETERMINED


def test_lid176_codes_split_hindi_and_urdu_by_script():
    assert to_label("en", "hello") == "eng_Latn"
    assert to_label("ms", "apa khabar") == "zsm_Latn"
    assert to_label("hi", "aap kaise hain") == "hin_Latn"
    assert to_label("hi", "आप कैसे हैं") == "hin_Deva"
    assert to_label("ur", "aap kaisay hain") == "urd_Latn"
    assert to_label("ur", "آپ کیسے ہیں") == "other"


def test_everything_else_is_other():
    assert to_label("yue_Hani", "你哋好") == "other"
    assert to_label("urd_Arab", "آپ کیسے ہیں") == "other"
    assert to_label("fr", "bonjour") == "other"


def test_scorer_detector_normalizes_then_maps():
    detector = ScorerDetector("glotlid", FakeScorer([("cmn_Hani", 0.9)]))
    assert detector.predict_many(["如何制作\n软气动执行器"]) == [("zho_Hans", 0.9)]


def test_empty_scorer_output_is_undetermined():
    assert ScorerDetector("x", FakeScorer([])).predict_many(["text"]) == [(UNDETERMINED, 0.0)]


def test_scorer_detector_uses_batches_when_offered():
    class BatchScorer:
        def __init__(self):
            self.sizes = []

        def top(self, text, k):
            raise AssertionError("top_many should be used")

        def top_many(self, texts, k):
            self.sizes.append(len(texts))
            return [[FakeScorer([("eng_Latn", 0.8)]).top(t, k)[0]] for t in texts]

    scorer = BatchScorer()
    assert len(ScorerDetector("batch", scorer).predict_many(["hello"] * 300)) == 300
    assert scorer.sizes == [256, 44]


def test_two_stage_adapter_reports_the_rules_uncertainty_as_undetermined():
    adapter = TwoStageAdapter("two-stage", TwoStageDetector(FakeScorer([("cmn_Hani", 0.95)]), 0.6))
    assert adapter.predict_many(["中文"]) == [(UNDETERMINED, 0.95)]


def test_specs_list_only_models_that_exist(tmp_path):
    assert detector_specs(tmp_path, include_lid176=False) == []
    (tmp_path / "glotlid-q.ftz").write_bytes(b"")
    for name in ("classifier.onnx", "tokenizer.json"):
        shutil.copy(SPECIALIST_FIXTURES / name, tmp_path / name)
    shutil.copy(SPECIALIST_FIXTURES / "classifier.onnx", tmp_path / "lid-specialist-e5-finetune.onnx")
    shutil.copy(SPECIALIST_FIXTURES / "classifier.json", tmp_path / "lid-specialist-e5-finetune.json")
    specs = {spec.name: spec for spec in detector_specs(tmp_path)}
    assert list(specs) == [
        "glotlid-compressed",
        "lid.176",
        "specialist-e5-finetune",
        "two-stage+specialist-e5-finetune",
    ]
    alone = specs["specialist-e5-finetune"].build()
    assert alone.predict_many(["lembut lembut"])[0][0] == "zsm_Latn"
