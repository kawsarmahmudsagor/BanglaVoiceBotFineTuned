# Build context is backend/ (the uv workspace root: pyproject.toml and uv.lock live there).
FROM ghcr.io/astral-sh/uv:0.9.7-python3.12-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && apt-get clean

# uid 1000 matches the default first user on Ubuntu, so files written under DATA_ROOT stay owned by the VM user.
RUN useradd --create-home --uid 1000 mlops

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"

# The workspace root references its members, so the fake backend's manifest must be present.
COPY pyproject.toml uv.lock .python-version ./
COPY app ./app
COPY workers ./workers
COPY alembic ./alembic
COPY alembic.ini ./
RUN uv sync --frozen --no-dev --extra server

USER mlops
EXPOSE 8000
CMD ["uvicorn", "app.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
