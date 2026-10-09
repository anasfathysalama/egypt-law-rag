"""Run the /ask API. Uses API_HOST and API_PORT from .env."""

import uvicorn

from egylaw_rag.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "egylaw_rag.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
    )


if __name__ == "__main__":
    main()
