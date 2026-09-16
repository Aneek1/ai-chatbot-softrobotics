# The frontend is built on the builder's own architecture: it emits the same files either way, and
# building it under emulation for arm64 would cost minutes for nothing.
# BUILDPLATFORM is set by buildx; the default keeps a plain docker build working without it.
ARG BUILDPLATFORM=linux/amd64
FROM --platform=$BUILDPLATFORM node:24-slim@sha256:2fe369e969550cde8e867afc3fe370b260140cab4a23d467074295b42163d553 AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build && node scripts/check-bundle.ts dist

FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea AS runtime
COPY --from=ghcr.io/astral-sh/uv:0.12.6 /uv /usr/local/bin/uv

ENV PYTHONUTF8=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=/usr/local/bin/python3.12 \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/app/.venv/bin:$PATH \
    MODELS_DIR=/models \
    INDEX_DIR=/data/index \
    HISTORY_DB=/data/history.db \
    FRONTEND_DIST=/app/frontend/dist

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

COPY backend/ backend/
COPY ingest/ ingest/
COPY scripts/ scripts/
COPY --from=frontend /frontend/dist frontend/dist

# Model files and the index are mounted, never copied: they are 3.2 GB and are fetched separately
# by the download script on the host before the image is run.
RUN useradd --create-home --uid 10001 app && mkdir -p /data /models && chown -R app:app /app /data
USER app
VOLUME ["/models", "/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health').read()"]
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
