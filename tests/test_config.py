from backend.app.config import REPO_ROOT, Settings


def test_defaults_point_inside_the_repo(monkeypatch):
    for key in ("GOOGLE_API_KEY", "GEMINI_MODEL", "GEMINI_TIMEOUT", "OLLAMA_URL", "OLLAMA_MODEL",
                "OLLAMA_TIMEOUT", "LANGID_MIN_CONFIDENCE", "RETRIEVAL_TOP_K", "MODELS_DIR", "INDEX_DIR"):
        monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None)
    assert settings.models_dir == REPO_ROOT / "models"
    assert settings.ollama_model == "qwen3:8b"
    assert settings.ollama_timeout == 120.0
    assert settings.gemini_timeout == 120.0
    assert settings.langid_min_confidence == 0.3


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


def test_api_key_is_not_printed(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "secret-value")
    settings = Settings(_env_file=None)
    assert "secret-value" not in repr(settings)
    assert settings.google_api_key.get_secret_value() == "secret-value"


def test_env_file_is_read_from_the_repo_root():
    assert Settings.model_config["env_file"] == REPO_ROOT / ".env"


def test_specialist_is_off_unless_configured(monkeypatch):
    monkeypatch.delenv("LANGID_SPECIALIST", raising=False)
    assert Settings(_env_file=None).langid_specialist is None
    monkeypatch.setenv("LANGID_SPECIALIST", "lid-specialist-fasttext.ftz")
    assert Settings(_env_file=None).langid_specialist == "lid-specialist-fasttext.ftz"
