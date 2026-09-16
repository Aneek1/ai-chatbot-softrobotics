from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    models_dir: Path = REPO_ROOT / "models"
    index_dir: Path = REPO_ROOT / "data" / "index"
    frontend_dist: Path = REPO_ROOT / "frontend" / "dist"
    glotlid_file: str = "glotlid-q.ftz"
    e5_dir: str = "multilingual-e5-small"

    langid_specialist: str | None = None  # file in models_dir written by training/export.py
    # smallest threshold with sentence accuracy >= 0.95 in results/langid-eval-2026-09-15.json
    langid_min_confidence: float = 0.3
    retrieval_top_k: int = 6

    google_api_key: SecretStr | None = None
    gemini_model: str | None = None
    gemini_timeout: float = 120.0

    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_timeout: float = 120.0

    private_mode: bool = False
    private_proxy: str | None = None
    # Longer than an Ollama answer may take, so a switch normally waits for it instead of failing.
    mode_switch_timeout: float = 150.0

    web_search: Literal["google", "duckduckgo", "off"] = "google"
    google_search_key: SecretStr | None = None
    google_search_engine_id: str | None = None
    search_timeout: float = 10.0
    search_max_results: int = 5

    history_db: Path = REPO_ROOT / "data" / "history.db"

    @field_validator("private_proxy")
    @classmethod
    def _proxy_must_resolve_names_itself(cls, value: str | None) -> str | None:
        if not value:
            return None
        parts = urlsplit(value)
        try:
            port = parts.port
        except ValueError:
            port = None
        # socks5:// would look up the search host with local DNS, which the proxy is meant to hide.
        if parts.scheme != "socks5h" or not parts.hostname or port is None:
            raise ValueError("PRIVATE_PROXY must look like socks5h://host:port")
        return value

    @property
    def gemini_configured(self) -> bool:
        return bool(self.google_api_key and self.google_api_key.get_secret_value() and self.gemini_model)

    @property
    def google_search_configured(self) -> bool:
        return bool(
            self.google_search_key
            and self.google_search_key.get_secret_value()
            and self.google_search_engine_id
        )
