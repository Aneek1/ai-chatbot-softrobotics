from fastapi.testclient import TestClient

from backend.app.config import REPO_ROOT, Settings
from backend.app.main import create_app
from tests.api_fakes import fake_services


def built_frontend(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text(
        "<!doctype html><title>Assistant</title>", encoding="utf-8", newline="\n"
    )
    (tmp_path / "assets" / "index-abc.js").write_text("console.log(1)\n", encoding="utf-8", newline="\n")
    return tmp_path


def test_the_page_is_served_at_the_root(tmp_path):
    with TestClient(create_app(fake_services(), frontend_dist=built_frontend(tmp_path))) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "<title>Assistant</title>" in response.text


def test_assets_are_served_with_their_own_bytes(tmp_path):
    with TestClient(create_app(fake_services(), frontend_dist=built_frontend(tmp_path))) as client:
        response = client.get("/assets/index-abc.js")
    assert response.status_code == 200
    assert response.text == "console.log(1)\n"


def test_an_unknown_path_returns_the_page_so_a_deep_link_works(tmp_path):
    with TestClient(create_app(fake_services(), frontend_dist=built_frontend(tmp_path))) as client:
        response = client.get("/chats/1234")
    assert response.status_code == 200
    assert "<title>Assistant</title>" in response.text


def test_the_api_is_not_shadowed_by_the_mount(tmp_path):
    with TestClient(create_app(fake_services(), frontend_dist=built_frontend(tmp_path))) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["detector"] == "fake"


def test_nothing_is_mounted_without_a_build(tmp_path):
    with TestClient(create_app(fake_services(), frontend_dist=tmp_path / "missing")) as client:
        response = client.get("/")
    assert response.status_code == 404


def test_the_default_build_directory_is_the_vite_output():
    settings = Settings(_env_file=None, private_proxy=None)
    assert settings.frontend_dist == REPO_ROOT / "frontend" / "dist"
