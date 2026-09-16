import threading

from backend.pipeline.langid import Candidate, Detection
from backend.store.history import Exchange, SqliteHistory

DETECTION = {
    "candidates": [{"code": "ind_Latn", "name": "Indonesian", "script": "Latn", "probability": 0.97}],
    "chosen": "ind_Latn",
    "chosen_name": "Indonesian",
    "uncertain": False,
    "stage": "general",
}


def exchange(question="Bagaimana cara membuat aktuator?", answer="Jawaban [1]"):
    return Exchange(
        question=question,
        detection=DETECTION,
        answer=answer,
        answer_language="ind_Latn",
        sources=[{"id": "d1:0", "title": "T1"}],
        web=[{"title": "W", "url": "https://example.org", "snippet": "s", "engine": "duckduckgo"}],
        citations={"valid": [1], "removed": []},
    )


class Clock:
    def __init__(self):
        self.tick = 0

    def __call__(self):
        self.tick += 1
        return f"2026-09-16T00:00:{self.tick:02d}.000+00:00"


def test_exchange_is_stored_as_two_messages(tmp_path):
    store = SqliteHistory.open(tmp_path / "history.db", clock=Clock())
    store.add_exchange("c1", exchange())
    chat = store.get_chat("c1")
    store.close()
    assert chat == {
        "id": "c1",
        "title": "Bagaimana cara membuat aktuator?",
        "created_at": "2026-09-16T00:00:01.000+00:00",
        "updated_at": "2026-09-16T00:00:01.000+00:00",
        "messages": [
            {
                "role": "user",
                "content": "Bagaimana cara membuat aktuator?",
                "created_at": "2026-09-16T00:00:01.000+00:00",
                "language": DETECTION,
            },
            {
                "role": "assistant",
                "content": "Jawaban [1]",
                "created_at": "2026-09-16T00:00:01.000+00:00",
                "answer_language": "ind_Latn",
                "sources": [{"id": "d1:0", "title": "T1"}],
                "web": [{"title": "W", "url": "https://example.org", "snippet": "s", "engine": "duckduckgo"}],
                "citations": {"valid": [1], "removed": []},
            },
        ],
    }


def test_history_survives_reopening_and_creates_the_folder(tmp_path):
    path = tmp_path / "data" / "history.db"
    store = SqliteHistory.open(path)
    store.add_exchange("c1", exchange(question="Tamil: மென்மையான ரோபோ"))
    store.close()

    reopened = SqliteHistory.open(path)
    assert [c["title"] for c in reopened.list_chats()] == ["Tamil: மென்மையான ரோபோ"]
    reopened.close()


def test_chats_are_listed_by_latest_activity():
    store = SqliteHistory.in_memory(clock=Clock())
    store.add_exchange("c1", exchange(question="first"))
    store.add_exchange("c2", exchange(question="second"))
    store.add_exchange("c1", exchange(question="third"))
    chats = store.list_chats()
    assert [(c["id"], c["title"]) for c in chats] == [("c1", "first"), ("c2", "second")]
    assert set(chats[0]) == {"id", "title", "created_at", "updated_at"}
    assert chats[0]["created_at"] < chats[0]["updated_at"]


def test_title_is_the_first_question_on_one_line_and_shortened():
    store = SqliteHistory.in_memory()
    store.add_exchange("c1", exchange(question="line one\nline two " + "x" * 200))
    title = store.list_chats()[0]["title"]
    assert title.startswith("line one line two x")
    assert len(title) == 80


def test_detections_come_back_for_the_answer_language_rule():
    store = SqliteHistory.in_memory()
    store.add_exchange("c1", exchange())
    store.add_exchange("c1", exchange())
    expected = Detection((Candidate("ind_Latn", 0.97),), "ind_Latn", False, "general")
    assert store.detections("c1") == [expected, expected]
    assert store.detections("unknown") == []


def test_delete_removes_the_chat_and_its_messages():
    store = SqliteHistory.in_memory()
    store.add_exchange("c1", exchange())
    assert store.delete_chat("c1") is True
    assert store.get_chat("c1") is None
    assert store.delete_chat("c1") is False

    store.add_exchange("c1", exchange())
    assert len(store.get_chat("c1")["messages"]) == 2


def test_in_memory_store_writes_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = SqliteHistory.in_memory()
    store.add_exchange("c1", exchange())
    assert len(store.list_chats()) == 1
    store.close()
    assert list(tmp_path.iterdir()) == []


def test_clear_removes_every_chat():
    store = SqliteHistory.in_memory()
    store.add_exchange("c1", exchange())
    store.add_exchange("c2", exchange())
    store.clear()
    assert store.list_chats() == []
    assert store.detections("c1") == []


def test_unknown_chat_is_none():
    assert SqliteHistory.in_memory().get_chat("missing") is None


def test_writes_from_several_threads():
    store = SqliteHistory.in_memory()

    def work(chat_id):
        for _ in range(20):
            store.add_exchange(chat_id, exchange())

    threads = [threading.Thread(target=work, args=(f"c{n}",)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(store.list_chats()) == 4
    assert all(len(store.get_chat(f"c{n}")["messages"]) == 40 for n in range(4))
