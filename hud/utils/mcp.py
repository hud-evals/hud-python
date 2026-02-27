from __future__ import annotations


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
