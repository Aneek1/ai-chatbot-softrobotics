from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.privacy.mode import ModeState
from backend.store.history import Exchange
from tests.api_fakes import fake_services

DETECTION = {
    "candidates": [],
    "chosen": "eng_Latn",
    "chosen_name": "English",
    "uncertain": False,
    "stage": "general",
}


def exchange(question):
    return Exchange(question, DETECTION, "answer", "eng_Latn", [], [], {"valid": [], "removed": []})


def test_list_get_and_delete():
    services = fake_services()
    with TestClient(create_app(services)) as client:
        services.history.add_exchange("c1", exchange("first question"))
        listed = client.get("/api/chats").json()
        chat = client.get("/api/chats/c1").json()
        deleted = client.delete("/api/chats/c1")
        missing = client.get("/api/chats/c1")
        deleted_again = client.delete("/api/chats/c1")
    assert [(c["id"], c["title"]) for c in listed] == [("c1", "first question")]
    assert [m["role"] for m in chat["messages"]] == ["user", "assistant"]
    assert deleted.status_code == 204
    assert missing.status_code == 404
    assert deleted_again.status_code == 404


def test_answered_chat_appears_with_its_messages():
    with TestClient(create_app(fake_services())) as client:
        client.post("/api/chat", json={"message": "소프트 로봇", "chat_id": "c9"})
        listed = client.get("/api/chats").json()
        chat = client.get("/api/chats/c9").json()
    assert [c["id"] for c in listed] == ["c9"]
    assert [(m["role"], m["content"]) for m in chat["messages"]] == [
        ("user", "소프트 로봇"),
        ("assistant", "안녕"),
    ]
    assert chat["messages"][0]["language"]["chosen"] == "kor_Hang"


def test_private_mode_shows_only_private_chats():
    services = fake_services(mode=ModeState(private=True))
    with TestClient(create_app(services)) as client:
        services.history.add_exchange("c1", exchange("normal question"))
        services.private_history.add_exchange("p1", exchange("private question"))
        listed = client.get("/api/chats").json()
        normal_chat = client.get("/api/chats/c1")
    assert [c["id"] for c in listed] == ["p1"]
    assert normal_chat.status_code == 404
