from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
DOCKERIGNORE = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def test_secrets_models_and_local_data_stay_out_of_the_build_context():
    for entry in (".env", "models/", "data/", ".git/", "spacy_env/", ".venv/"):
        assert entry in DOCKERIGNORE, entry


def test_the_build_downloads_no_model():
    assert "download_models" not in DOCKERFILE
    assert "HF_HUB_OFFLINE=1" in DOCKERFILE


def test_both_base_images_are_pinned_by_digest():
    froms = [line for line in DOCKERFILE.splitlines() if line.startswith("FROM ")]
    assert len(froms) == 2
    assert all("@sha256:" in line for line in froms)
    assert "COPY --from=ghcr.io/astral-sh/uv:0.12.6" in DOCKERFILE


def test_the_image_serves_the_built_frontend_as_an_unprivileged_user():
    assert "COPY --from=frontend /frontend/dist frontend/dist" in DOCKERFILE
    assert "FRONTEND_DIST=/app/frontend/dist" in DOCKERFILE
    assert "USER app" in DOCKERFILE


def test_compose_points_the_app_at_the_ollama_service_and_mounts_models_read_only():
    app = COMPOSE["services"]["app"]
    assert app["environment"]["OLLAMA_URL"] == "http://ollama:11434"
    assert "./models:/models:ro" in app["volumes"]
    assert "ollama" in COMPOSE["services"]


def test_compose_keeps_the_index_and_the_history_in_a_named_volume():
    assert "app-data:/data" in COMPOSE["services"]["app"]["volumes"]
    assert "app-data" in COMPOSE["volumes"]
