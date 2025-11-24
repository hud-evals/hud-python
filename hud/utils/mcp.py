from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from hud.settings import settings

logger = logging.getLogger(__name__)


class MCPConfigPatch(BaseModel):
    """Patch for MCP config."""

    headers: dict[str, Any] | None = Field(default_factory=dict, alias="headers")
    meta: dict[str, Any] | None = Field(default_factory=dict, alias="meta")


def _is_hud_server(url: str) -> bool:
    """Check if a URL is a HUD MCP server.

    Matches any mcp.hud.* domain (including .ai, .so, and future domains).
    """
    if not url:
        return False
    return "mcp.hud." in url.lower()


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


def setup_hud_telemetry(
    mcp_config: dict[str, dict[str, Any]], auto_trace: bool = True
) -> Any | None:
    """Setup telemetry for hud servers.

    Returns:
        The auto-created trace context manager if one was created, None otherwise.
        Caller is responsible for exiting the context manager.
    """
    if mcp_config is None:
        raise ValueError("Please run initialize() before setting up client-side telemetry")

    # Check if there are any HUD servers to setup telemetry for
    has_hud_servers = any(
        _is_hud_server(server_cfg.get("url", "")) for server_cfg in mcp_config.values()
    )

    # If no HUD servers, no need for telemetry setup
    if not has_hud_servers:
        return None

    from hud.otel import get_current_task_run_id
    from hud.telemetry import trace

    run_id = get_current_task_run_id()
    auto_trace_cm = None

    if not run_id and auto_trace:
        # Start an auto trace and capture its ID for headers/metadata
        auto_trace_cm = trace("My Trace")
        _trace_obj = auto_trace_cm.__enter__()
        try:
            run_id = getattr(_trace_obj, "id", None) or str(_trace_obj)
        except Exception:  # pragma: no cover - fallback shouldn't fail lint
            run_id = None

    # Patch HUD servers with run-id (works whether auto or user trace)
    if run_id:
        patch_mcp_config(
            mcp_config,
            MCPConfigPatch(headers={"Run-Id": run_id}, meta={"run_id": run_id}),
        )

    if settings.api_key:
        patch_mcp_config(
            mcp_config,
            MCPConfigPatch(headers={"Authorization": f"Bearer {settings.api_key}"}),
        )

    return auto_trace_cm
