from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    models_dir: Path = REPO_ROOT / "models"
    index_dir: Path = REPO_ROOT / "data" / "index"
    glotlid_file: str = "glotlid-q.ftz"
    e5_dir: str = "multilingual-e5-small"

    langid_min_confidence: float = 0.60
    retrieval_top_k: int = 6

    google_api_key: str | None = None
    gemini_model: str | None = None

    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_timeout: float = 120.0

    @property
    def gemini_configured(self) -> bool:
        return bool(self.google_api_key and self.gemini_model)
