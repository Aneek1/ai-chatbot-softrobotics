import os

# Models are downloaded ahead of time; the running app must never fetch from Hugging Face.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI  # noqa: E402

from backend.app.api import chat, health  # noqa: E402
from backend.app.config import Settings  # noqa: E402
from backend.app.services import Services, build_services  # noqa: E402


def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.services is None:
            app.state.services = build_services(Settings())
        yield
        if app.state.services.index is not None:
            app.state.services.index.close()

    app = FastAPI(title="Soft-robotics assistant", lifespan=lifespan)
    app.state.services = services
    app.include_router(chat.router)
    app.include_router(health.router)
    return app


app = create_app()
