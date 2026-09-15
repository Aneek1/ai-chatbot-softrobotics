import backend.app.services as services_module
from backend.app.config import Settings
from backend.app.services import build_services
from tests.fakes import FakeEmbedder, FakeScorer


class FakeIndex:
    @staticmethod
    def open(path, embedder):
        return FakeIndex()

    def close(self):
        pass


def patch_models(monkeypatch) -> list:
    loaded = []

    def fake_load(models_dir, name, embedder=None):
        loaded.append((name, embedder))
        return FakeScorer([("zsm_Latn", 0.9), ("ind_Latn", 0.1)])

    monkeypatch.setattr(
        services_module, "FastTextScorer", lambda path: FakeScorer([("ind_Latn", 0.7), ("zsm_Latn", 0.3)])
    )
    monkeypatch.setattr(services_module, "E5Embedder", lambda path: FakeEmbedder())
    monkeypatch.setattr(services_module, "ChunkIndex", FakeIndex)
    monkeypatch.setattr(services_module, "load_specialist", fake_load)
    return loaded


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, google_api_key=None, gemini_model=None, **overrides)


def test_no_specialist_by_default(monkeypatch):
    loaded = patch_models(monkeypatch)
    services = build_services(settings(langid_specialist=None))
    assert loaded == []
    assert services.detector_name == "glotlid-q.ftz"
    assert services.chat.detector.detect("Bagaimana cara membuat").stage == "general"


def test_empty_setting_means_no_specialist(monkeypatch):
    loaded = patch_models(monkeypatch)
    build_services(settings(langid_specialist=""))
    assert loaded == []


def test_configured_specialist_becomes_stage_two(monkeypatch):
    loaded = patch_models(monkeypatch)
    services = build_services(settings(langid_specialist="lid-specialist-e5-head.onnx"))
    assert loaded[0][0] == "lid-specialist-e5-head.onnx"
    assert isinstance(loaded[0][1], FakeEmbedder)
    assert services.detector_name == "glotlid-q.ftz + lid-specialist-e5-head.onnx"
    result = services.chat.detector.detect("Bagaimana cara membuat")
    assert result.chosen == "zsm_Latn"
    assert result.stage == "specialist"
