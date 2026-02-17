from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MCPConfigPatch(BaseModel):
    """Patch for MCP config."""

    headers: dict[str, Any] | None = Field(default_factory=dict, alias="headers")
    meta: dict[str, Any] | None = Field(default_factory=dict, alias="meta")


def _is_hud_server(url: str) -> bool:
    """Check if a URL is a HUD MCP server.

    Matches:
    - Any mcp.hud.* domain (including .ai, .so, and future domains)
    - Staging servers (orcstaging.hud.so)
    - Any *.hud.ai or *.hud.so domain
    """
    if not url:
        return False
    url_lower = url.lower()
    return "mcp.hud." in url_lower or ".hud.ai" in url_lower or ".hud.so" in url_lower


def patch_mcp_config(mcp_config: dict[str, dict[str, Any]], patch: MCPConfigPatch) -> None:
    """Patch MCP config with additional values."""
    for server_cfg in mcp_config.values():
        url = server_cfg.get("url", "")

        # 1) HTTP header lane (only for hud MCP servers)
        if _is_hud_server(url) and patch.headers:
            for key, value in patch.headers.items():
                headers = server_cfg.setdefault("headers", {})
                headers.setdefault(key, value)

        # 2) Metadata lane (for all servers)
        if patch.meta:
            for key, value in patch.meta.items():
                meta = server_cfg.setdefault("meta", {})
                meta.setdefault(key, value)
