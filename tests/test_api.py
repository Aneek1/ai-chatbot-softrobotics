import json

from fastapi.testclient import TestClient

from backend.app.chat_service import ChatService
from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.services import Services
from backend.pipeline.langid import TwoStageDetector
from tests.fakes import FakeModel, FakeScorer


class EmptyRetriever:
    def search(self, query, top_k):
        return []


def fake_services():
    chat = ChatService(
        detector=TwoStageDetector(general=FakeScorer([("kor_Hang", 0.99)]), min_confidence=0.6),
        retriever=EmptyRetriever(),
        models={"ollama": FakeModel(pieces=("안녕",))},
        top_k=6,
    )
    return Services(
        settings=Settings(_env_file=None, google_api_key=None, gemini_model=None),
        chat=chat,
        detections={},
        ollama=None,
        index=None,
        detector_name="fake",
    )


def parse_sse(body: str):
    events = []
    for block in body.strip().split("\n\n"):
        if block.startswith(":"):
            continue  # keep-alive comment
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_chat_streams_server_sent_events():
    with TestClient(create_app(fake_services())) as client:
        response = client.post("/api/chat", json={"message": "소프트 로봇", "model": "ollama"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [e for e, _ in events] == ["language", "sources", "notice", "token", "citations", "done"]
    assert events[0][1]["chosen"] == "kor_Hang"


def test_chat_id_keeps_detection_history():
    services = fake_services()
    with TestClient(create_app(services)) as client:
        client.post("/api/chat", json={"message": "소프트 로봇", "chat_id": "c1"})
        client.post("/api/chat", json={"message": "로봇", "chat_id": "c1"})
    assert len(services.detections["c1"]) == 2


def test_empty_message_is_rejected():
    with TestClient(create_app(fake_services())) as client:
        response = client.post("/api/chat", json={"message": ""})
    assert response.status_code == 422


def test_health_without_ollama_or_index():
    with TestClient(create_app(fake_services())) as client:
        body = client.get("/api/health").json()
    assert body == {
        "detector": "fake",
        "ollama_reachable": False,
        "gemini_configured": False,
        "documents_by_language": {},
    }
