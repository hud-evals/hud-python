"""Sample browser task factory."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from hud.settings import settings
from hud.types import MCPToolCall, Task


class BrowserTask(Task):
    """Task subclass with browser defaults for BrowserTask(prompt=...)."""

    prompt: str = "Open Google and be ready to search."
    mcp_config: dict[str, Any] = Field(
        default_factory=lambda: {
            "browser": {
                "url": "https://mcp.hud.ai/v3/mcp",
                "headers": {
                    "Authorization": f"Bearer {settings.api_key}",
                    "Mcp-Image": "hudevals/hud-remote-browser:0.1.1",
                },
            }
        }
    )
    setup_tool: MCPToolCall | list[MCPToolCall] | None = Field(
        default_factory=lambda: MCPToolCall(
            name="setup",
            arguments={"name": "navigate_to_url", "arguments": {"url": "https://www.google.com"}},
        )
    )
