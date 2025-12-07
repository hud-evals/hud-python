from __future__ import annotations

import base64
import contextlib
import json
import logging
from typing import Any

from hud.settings import settings
from hud.shared.requests import make_request
from hud.utils.hud_console import hud_console

logger = logging.getLogger(__name__)


async def create_dynamic_trace(
    *,
    mcp_config: dict[str, dict[str, Any]],
    build_status: bool,
    environment_name: str,
) -> tuple[str | None, str | None]:
    """
    Create a dynamic trace for HUD dev sessions when running in HTTP mode.

    Sends a POST to the HUD API with:
      - mcp_config: points to the local MCP config (same as Cursor)
      - build_status: True if Docker mode (built image), False if basic Python mode
      - environment_name: Name of the environment/server/image
      - git_info: Repository information (if available)

    Returns the full URL to the live trace when successful, otherwise None.
    """
    api_base = settings.hud_api_url.rstrip("/")
    # Endpoint TBD; use a sensible default path that the backend can wire up
    url = f"{api_base}/dev/dynamic-traces"

    # Get git repository information
    from hud.cli.utils.git import get_git_info

    git_info = get_git_info()

    payload = {
        "mcp_config": mcp_config,
        "build_status": bool(build_status),
        "environment_name": environment_name,
    }

    # Add git info if available
    if git_info and git_info.get("remote_url"):
        payload["git_info"] = git_info
        logger.info("Detected git repository: %s", git_info.get("remote_url"))
    else:
        logger.info("No git repository detected")

    # Require API key for dev mode
    api_key = settings.api_key
    if not api_key:
        hud_console.error("HUD_API_KEY is required for hud dev command")
        hud_console.info("")
        hud_console.info("Please set your API key using one of these methods:")
        hud_console.info("  1. Set environment variable: export HUD_API_KEY=your_key")
        hud_console.info("  2. Use hud set command: hud set api_key your_key")
        hud_console.info("")
        hud_console.info("Get your API key at: https://hud.ai/settings")
        import sys

        sys.exit(1)

    try:
        resp = await make_request("POST", url=url, json=payload, api_key=api_key)
        # New API returns an id; construct the URL as https://hud.ai/trace/{id}
        trace_id = resp.get("id")

        if isinstance(trace_id, str) and trace_id:
            return trace_id, f"https://hud.ai/trace/{trace_id}"
        return None, None
    except Exception as e:
        # Do not interrupt dev flow
        try:
            preview = json.dumps(payload)[:500]
            logger.warning("Failed to create dynamic dev trace: %s | payload=%s", e, preview)
        except Exception:
            logger.warning("Failed to create dynamic dev trace: %s", e)
        return None, None


def show_dev_ui(
    *,
    live_trace_url: str,
    server_name: str,
    port: int,
    cursor_deeplink: str,
    is_docker: bool = False,
) -> None:
    """
    Show the minimal dev UI with live trace link.

    This is called only when we have a successful trace URL.
    For full UI mode, the caller should use show_dev_server_info() directly.

    Args:
        live_trace_url: URL to the live trace
        server_name: Name of the server/image
        port: Port the server is running on
        cursor_deeplink: Pre-generated Cursor deeplink URL
        is_docker: Whether this is Docker mode (affects hot-reload message)
    """
    import webbrowser

    from rich.panel import Panel

    # Show header first
    hud_console.header("HUD Development Server", icon="🚀")

    # Try to open the live trace in the default browser
    with contextlib.suppress(Exception):
        # new=2 -> open in a new tab, if possible
        webbrowser.open(live_trace_url, new=2)

    # Show panel with just the link
    # Center the link and style it: blue, bold, underlined
    link_markup = f"[bold underline rgb(108,113,196)][link={live_trace_url}]{live_trace_url}[/link][/bold underline rgb(108,113,196)]"  # noqa: E501
    # Use center alignment by surrounding with spaces via justify
    from rich.align import Align

    panel = Panel(
        Align.center(link_markup),
        title="🔗 Live Dev Trace",
        border_style="rgb(192,150,12)",  # HUD gold
        padding=(1, 2),
    )
    hud_console.console.print(panel)

    # Show other info below
    label = "Base image" if is_docker else "Server"
    hud_console.info("")
    hud_console.info(f"{hud_console.sym.ITEM} {label}: {server_name}")
    hud_console.info(f"{hud_console.sym.ITEM} Cursor:")
    # Display the Cursor link on its own line to prevent wrapping
    hud_console.link(cursor_deeplink)
    hud_console.info("")
    hud_console.info(f"{hud_console.sym.SUCCESS} Hot-reload enabled")
    if is_docker:
        hud_console.dim_info(
            "",
            "Container restarts on file changes (mounted volumes), "
            "if changing tools run hud dev again",
        )
    hud_console.info("")


def generate_cursor_deeplink(server_name: str, port: int) -> str:
    """Generate a Cursor deeplink for the MCP server.

    Args:
        server_name: Name of the server
        port: Port the server is running on

    Returns:
        Cursor deeplink URL
    """
    server_config = {"url": f"http://localhost:{port}/mcp"}
    config_json = json.dumps(server_config, indent=2)
    config_base64 = base64.b64encode(config_json.encode()).decode()
    return (
        f"cursor://anysphere.cursor-deeplink/mcp/install?name={server_name}&config={config_base64}"
    )
