import socket

import pytest
from fastapi.testclient import TestClient

import backend.app.main as main_module
from backend.app.config import Settings
from backend.app.main import create_app
from backend.privacy.egress import EgressBlocked, EgressGuard
from backend.privacy.egress_log import EgressLog
from backend.privacy.mode import ModeState
from backend.privacy.policy import EgressPolicy
from tests.api_fakes import fake_services


def test_app_installs_the_guard_for_its_lifetime(monkeypatch):
    looked_up = []
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, *args, **kwargs: looked_up.append(host) or [])
    mode, log = ModeState(), EgressLog()
    guard = EgressGuard(EgressPolicy(normal=frozenset(), private=frozenset()), lambda: mode.private, log)
    with TestClient(create_app(fake_services(mode=mode, egress_log=log, guard=guard))):
        assert guard.installed
        with pytest.raises(EgressBlocked):
            socket.getaddrinfo("example.com", 443)
    assert not guard.installed
    assert looked_up == []


def test_startup_installs_the_guard_before_services_load(monkeypatch, tmp_path):
    seen = []

    def fake_build_services(settings, privacy):
        seen.append(privacy.guard.installed)
        return fake_services(mode=privacy.mode, egress_log=privacy.egress_log, guard=privacy.guard)

    settings = Settings(_env_file=None, google_api_key=None, gemini_model=None, history_db=tmp_path / "h.db")
    monkeypatch.setattr(main_module, "Settings", lambda: settings)
    monkeypatch.setattr(main_module, "build_services", fake_build_services)
    app = create_app()
    with TestClient(app):
        assert app.state.services.guard.installed
    assert seen == [True]
    assert not app.state.services.guard.installed


def test_failed_startup_removes_the_guard(monkeypatch, tmp_path):
    def broken_build_services(settings, privacy):
        raise FileNotFoundError("Language-ID model not found")

    settings = Settings(_env_file=None, google_api_key=None, gemini_model=None, history_db=tmp_path / "h.db")
    monkeypatch.setattr(main_module, "Settings", lambda: settings)
    monkeypatch.setattr(main_module, "build_services", broken_build_services)
    with pytest.raises(FileNotFoundError), TestClient(create_app()):
        pass


def test_health_reports_search_mode_guard_and_model_files(tmp_path):
    # Only the language-ID model is on disk, so each file is reported on its own.
    (tmp_path / "glotlid-q.ftz").write_bytes(b"")
    settings = Settings(
        _env_file=None,
        google_api_key=None,
        gemini_model=None,
        private_proxy=None,
        models_dir=tmp_path,
        langid_specialist="specialist.onnx",
    )
    services = fake_services(settings=settings, mode=ModeState(private=True))
    with TestClient(create_app(services)) as client:
        body = client.get("/api/health").json()
    assert body == {
        "detector": "fake",
        "model_files": {"glotlid": True, "e5": False, "specialist": False},
        "ollama_reachable": False,
        "gemini_configured": False,
        "web_search": {"normal": None, "private": None},
        "private": True,
        "egress_guard": False,
        "documents_by_language": {},
    }
