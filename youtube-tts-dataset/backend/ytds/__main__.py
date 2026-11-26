"""`python -m ytds`: serve the API and UI on 127.0.0.1:<YTDS_PORT>."""
from __future__ import annotations

import uvicorn

from .api import create_app
from .config import load_settings


def main() -> None:
    settings = load_settings()
    uvicorn.run(create_app(settings), host="127.0.0.1", port=settings.port)


if __name__ == "__main__":
    main()
