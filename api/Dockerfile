# Build context is the repo root (uv workspace + uv.lock live there).
FROM ghcr.io/astral-sh/uv:0.9.7-python3.12-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && apt-get clean

# uid 1000 matches the default first user on Ubuntu, so files written under DATA_ROOT stay owned by the VM user.
RUN useradd --create-home --uid 1000 mlops

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"

# The workspace root references every member, so all member manifests must be present.
COPY pyproject.toml uv.lock .python-version ./
COPY shared ./shared
COPY api ./api
COPY worker ./worker
COPY backends/fake ./backends/fake
RUN uv sync --frozen --no-dev --package mlops-api

USER mlops
EXPOSE 8000
CMD ["uvicorn", "mlops_api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
