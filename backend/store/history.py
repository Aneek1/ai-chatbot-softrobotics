import json
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.pipeline.langid import Candidate, Detection

TITLE_CHARS = 80

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_by_chat ON messages (chat_id, id);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class Exchange:
    question: str
    detection: dict[str, Any]
    answer: str
    answer_language: str
    sources: list[dict[str, Any]]
    web: list[dict[str, Any]]
    citations: dict[str, list[int]]


def detection_from_payload(payload: dict[str, Any]) -> Detection:
    return Detection(
        candidates=tuple(Candidate(c["code"], c["probability"]) for c in payload["candidates"]),
        chosen=payload["chosen"],
        uncertain=payload["uncertain"],
        stage=payload["stage"],
    )


class SqliteHistory:
    """Chat history in SQLite: a file for normal mode, or memory only for private mode."""

    def __init__(self, connection: sqlite3.Connection, clock: Callable[[], str] = _now):
        self._db = connection
        self._lock = threading.Lock()
        self._clock = clock
        with self._lock:
            self._db.execute("PRAGMA foreign_keys = ON")
            self._db.executescript(_SCHEMA)

    @classmethod
    def open(cls, path: Path, clock: Callable[[], str] = _now) -> "SqliteHistory":
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(sqlite3.connect(path, check_same_thread=False), clock)

    @classmethod
    def in_memory(cls, clock: Callable[[], str] = _now) -> "SqliteHistory":
        return cls(sqlite3.connect(":memory:", check_same_thread=False), clock)

    def add_exchange(self, chat_id: str, exchange: Exchange) -> None:
        now = self._clock()
        title = " ".join(exchange.question.split())[:TITLE_CHARS]
        user = {"language": exchange.detection}
        assistant = {
            "answer_language": exchange.answer_language,
            "sources": exchange.sources,
            "web": exchange.web,
            "citations": exchange.citations,
        }
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at",
                (chat_id, title, now, now),
            )
            self._db.executemany(
                "INSERT INTO messages (chat_id, role, content, created_at, data) VALUES (?, ?, ?, ?, ?)",
                [
                    (chat_id, "user", exchange.question, now, json.dumps(user, ensure_ascii=False)),
                    (chat_id, "assistant", exchange.answer, now, json.dumps(assistant, ensure_ascii=False)),
                ],
            )

    def list_chats(self) -> list[dict[str, str]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, title, created_at, updated_at FROM chats ORDER BY updated_at DESC, rowid DESC"
            ).fetchall()
        return [{"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]} for r in rows]

    def get_chat(self, chat_id: str) -> dict[str, Any] | None:
        with self._lock:
            chat = self._db.execute(
                "SELECT id, title, created_at, updated_at FROM chats WHERE id = ?", (chat_id,)
            ).fetchone()
            if chat is None:
                return None
            rows = self._db.execute(
                "SELECT role, content, created_at, data FROM messages WHERE chat_id = ? ORDER BY id",
                (chat_id,),
            ).fetchall()
        messages = [{"role": r[0], "content": r[1], "created_at": r[2], **json.loads(r[3])} for r in rows]
        return {
            "id": chat[0],
            "title": chat[1],
            "created_at": chat[2],
            "updated_at": chat[3],
            "messages": messages,
        }

    def detections(self, chat_id: str) -> list[Detection]:
        with self._lock:
            rows = self._db.execute(
                "SELECT data FROM messages WHERE chat_id = ? AND role = 'user' ORDER BY id", (chat_id,)
            ).fetchall()
        return [detection_from_payload(json.loads(r[0])["language"]) for r in rows]

    def delete_chat(self, chat_id: str) -> bool:
        with self._lock, self._db:
            return self._db.execute("DELETE FROM chats WHERE id = ?", (chat_id,)).rowcount > 0

    def clear(self) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM chats")

    def close(self) -> None:
        with self._lock:
            self._db.close()
