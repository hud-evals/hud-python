from __future__ import annotations

import contextlib
import logging
import sys
import time

import httpx

from hud.tools import PlaywrightTool

logger = logging.getLogger(__name__)

ENV_SERVER_URL = "http://localhost:8000"
http_client = httpx.AsyncClient(
    base_url=ENV_SERVER_URL, timeout=30.0, headers={"User-Agent": "HUD-Browser-Server/1.0"}
)


def _discover_cdp_url(timeout_sec: float = 60.0, poll_interval_sec: float = 0.5) -> str | None:
    """Synchronously poll the environment for a CDP websocket URL.

    Blocks import until CDP is available or times out. Ensures nothing is
    written to stdout to avoid corrupting stdio MCP transport.
    """
    deadline = time.time() + timeout_sec
    with contextlib.redirect_stdout(sys.stderr):
        try:
            with httpx.Client(base_url=ENV_SERVER_URL, timeout=5.0) as client:
                while time.time() < deadline:
                    try:
                        resp = client.get("/cdp")
                        if resp.status_code == 200:
                            ws = resp.json().get("ws")
                            if ws:
                                return ws
                    except Exception:
                        pass
                    time.sleep(poll_interval_sec)
                    logger.info("Polling for CDP URL")
        except Exception:
            raise
    return None


playwright = PlaywrightTool(cdp_url=_discover_cdp_url())

__all__ = ["ENV_SERVER_URL", "http_client", "playwright"]
