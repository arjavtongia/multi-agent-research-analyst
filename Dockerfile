FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first for better layer caching.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra ui

COPY app ./app
COPY data ./data

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000 8501

# API by default. For the UI:
#   docker run -p 8501:8501 --env-file .env IMAGE streamlit run app/streamlit_app.py --server.address 0.0.0.0
CMD ["research-agents", "serve", "--host", "0.0.0.0", "--port", "8000"]
