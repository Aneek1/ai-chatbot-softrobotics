import os

# Models are downloaded ahead of time; the running app must never fetch from Hugging Face.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from contextlib import asynccontextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

from fastapi import FastAPI  # noqa: E402

from backend.app.api import chat, chats, egress, health, mode  # noqa: E402
from backend.app.config import Settings  # noqa: E402
from backend.app.services import Services, build_privacy, build_services  # noqa: E402
from backend.app.static import mount_frontend  # noqa: E402


def create_app(services: Services | None = None, frontend_dist: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.services is None:
            settings = Settings()
            privacy = build_privacy(settings)
            # Installed before the models and the index load, so nothing at startup reaches the network.
            privacy.guard.install()
            try:
                app.state.services = build_services(settings, privacy)
            except BaseException:
                privacy.guard.uninstall()
                raise
        elif app.state.services.guard is not None:
            app.state.services.guard.install()
        services = app.state.services
        try:
            yield
        finally:
            if services.guard is not None:
                services.guard.uninstall()
            services.close()

    app = FastAPI(title="Soft-robotics assistant", lifespan=lifespan)
    app.state.services = services
    for router in (chat.router, chats.router, egress.router, health.router, mode.router):
        app.include_router(router)
    if frontend_dist is None:
        frontend_dist = services.settings.frontend_dist if services else Settings().frontend_dist
    # Mounted after the routers, so /api keeps winning over the catch-all.
    mount_frontend(app, frontend_dist)
    return app


app = create_app()
