FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src ./src
RUN uv sync --frozen --no-dev

COPY data/processed/civil_code.json data/processed/civil_code.json
COPY data/index data/index

EXPOSE 8000

CMD ["uvicorn", "egylaw_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
