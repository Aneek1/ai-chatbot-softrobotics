import json
import socket
import threading
import time
from contextlib import contextmanager

import uvicorn

from backend.app.chat_service import ChatService
from backend.app.config import Settings
from backend.app.services import Services
from backend.pipeline.langid import TwoStageDetector
from tests.fakes import FakeModel, FakeScorer


class EmptyRetriever:
    def search(self, query, top_k):
        return []


def fake_services(**overrides) -> Services:
    chat = ChatService(
        detector=TwoStageDetector(general=FakeScorer([("kor_Hang", 0.99)]), min_confidence=0.6),
        retriever=EmptyRetriever(),
        models={"ollama": FakeModel(pieces=("안녕",))},
        top_k=6,
    )
    values = {
        "settings": Settings(_env_file=None, google_api_key=None, gemini_model=None, private_proxy=None),
        "chat": chat,
        "ollama": None,
        "index": None,
        "detector_name": "fake",
    }
    return Services(**(values | overrides))


def parse_sse(body: str):
    events = []
    for block in body.strip().split("\n\n"):
        if not block or block.startswith(":"):
            continue  # keep-alive comment
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


@contextmanager
def serve(app):
    """Run the app on a free loopback port in a background thread; yields the base URL."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", timeout_graceful_shutdown=2))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
