"""Serve the built frontend next to the API, so the container runs one process."""

from pathlib import Path

from fastapi import FastAPI
from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope


class SpaFiles(StaticFiles):
    """Static files with one difference: an unknown path returns index.html instead of 404.

    The workspace is a single page, so a link to a chat has to reach it rather than the file server.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            return await super().get_response("index.html", scope)


def mount_frontend(app: FastAPI, dist: Path) -> bool:
    """Mount a built frontend at /. Returns whether there was one to mount."""
    if not (dist / "index.html").is_file():
        return False
    app.mount("/", SpaFiles(directory=dist, html=True), name="frontend")
    return True
