import pytest
from pydantic import ValidationError

from backend.app.config import REPO_ROOT, Settings


def test_defaults_point_inside_the_repo(monkeypatch):
    for key in (
        "GOOGLE_API_KEY",
        "GEMINI_MODEL",
        "GEMINI_TIMEOUT",
        "OLLAMA_URL",
        "OLLAMA_MODEL",
        "OLLAMA_TIMEOUT",
        "LANGID_MIN_CONFIDENCE",
        "RETRIEVAL_TOP_K",
        "MODELS_DIR",
        "INDEX_DIR",
    ):
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


PRIVACY_KEYS = (
    "PRIVATE_MODE",
    "PRIVATE_PROXY",
    "MODE_SWITCH_TIMEOUT",
    "WEB_SEARCH",
    "GOOGLE_SEARCH_KEY",
    "GOOGLE_SEARCH_ENGINE_ID",
    "SEARCH_TIMEOUT",
    "SEARCH_MAX_RESULTS",
    "HISTORY_DB",
)


def test_privacy_and_search_defaults(monkeypatch):
    for key in PRIVACY_KEYS:
        monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None)
    assert settings.private_mode is False
    assert settings.private_proxy is None
    assert settings.mode_switch_timeout == 150.0
    assert settings.web_search == "google"
    assert settings.google_search_configured is False
    assert settings.search_timeout == 10.0
    assert settings.search_max_results == 5
    assert settings.history_db == REPO_ROOT / "data" / "history.db"


def test_google_search_needs_key_and_engine_id(monkeypatch):
    monkeypatch.setenv("GOOGLE_SEARCH_KEY", "search-key")
    monkeypatch.delenv("GOOGLE_SEARCH_ENGINE_ID", raising=False)
    assert Settings(_env_file=None).google_search_configured is False

    monkeypatch.setenv("GOOGLE_SEARCH_ENGINE_ID", "engine-id")
    settings = Settings(_env_file=None)
    assert settings.google_search_configured is True
    assert "search-key" not in repr(settings)


def test_socks5h_proxy_is_accepted(monkeypatch):
    monkeypatch.setenv("PRIVATE_PROXY", "socks5h://127.0.0.1:9050")
    assert Settings(_env_file=None).private_proxy == "socks5h://127.0.0.1:9050"


def test_empty_proxy_means_no_proxy(monkeypatch):
    monkeypatch.setenv("PRIVATE_PROXY", "")
    assert Settings(_env_file=None).private_proxy is None


@pytest.mark.parametrize(
    "value",
    ["socks5://127.0.0.1:9050", "http://127.0.0.1:8080", "socks5h://127.0.0.1", "127.0.0.1:9050"],
)
def test_proxy_that_would_resolve_names_locally_or_lacks_a_port_is_rejected(monkeypatch, value):
    monkeypatch.setenv("PRIVATE_PROXY", value)
    with pytest.raises(ValidationError, match="socks5h://host:port"):
        Settings(_env_file=None)


def test_unknown_web_search_engine_is_rejected(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH", "bing")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
