from __future__ import annotations

import os
import httpx

# Environment port (as string to simplify formatting)
ENV_SERVER_PORT = os.getenv("ENV_SERVER_PORT", "8005")

# Shared HTTP client for talking to the environment backend
http_client = httpx.AsyncClient(
    base_url=f"http://localhost:{ENV_SERVER_PORT}",
    timeout=10.0,
)

__all__ = ["ENV_SERVER_PORT", "http_client"]
