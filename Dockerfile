# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project


FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    CHROMA_DIR="/app/data/chroma" \
    DOCS_DIR="/app/data/docs" \
    CATALOG_PATH="/app/data/catalog.sqlite3"

WORKDIR /app

# Crear usuario y grupo de ejecución sin privilegios
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/false -m -d /home/appuser appuser

COPY --from=builder /app/.venv /app/.venv
COPY app ./app
COPY scripts ./scripts
COPY eval ./eval
COPY README.md pyproject.toml ./

RUN mkdir -p /app/data/chroma /app/data/docs /app/data/demo_docs && \
    chown -R appuser:appgroup /app

VOLUME ["/app/data"]
EXPOSE 8000

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
