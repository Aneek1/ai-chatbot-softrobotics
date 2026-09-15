from backend.app.config import REPO_ROOT, Settings


def test_defaults_point_inside_the_repo(monkeypatch):
    for key in ("GOOGLE_API_KEY", "GEMINI_MODEL", "OLLAMA_MODEL"):
        monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None)
    assert settings.models_dir == REPO_ROOT / "models"
    assert settings.ollama_model == "qwen3:8b"
    assert settings.ollama_timeout == 120.0
    assert settings.langid_min_confidence == 0.60


def test_gemini_needs_both_key_and_model(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert Settings(_env_file=None).gemini_configured is False

    monkeypatch.setenv("GEMINI_MODEL", "some-model")
    assert Settings(_env_file=None).gemini_configured is True


def test_empty_values_do_not_configure_gemini(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("GEMINI_MODEL", "")
    assert Settings(_env_file=None).gemini_configured is False
