"""Typed file content for agent tools."""

from __future__ import annotations

import base64

from mcp.types import ImageContent

from hud.agents.tools.base import tool_err, tool_ok
from hud.types import MCPToolResult

#: Lines a file view shows when the model does not ask for a range.
DEFAULT_VIEW_LINES = 2000


def view_file(data: bytes) -> MCPToolResult:
    """Identify supported images by signature and preserve their encoded bytes."""
    mime_type = None
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime_type = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        mime_type = "image/jpeg"
    elif data.startswith((b"GIF87a", b"GIF89a")):
        mime_type = "image/gif"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        mime_type = "image/webp"
    if mime_type is not None:
        return MCPToolResult(
            content=[
                ImageContent(
                    type="image", mimeType=mime_type, data=base64.b64encode(data).decode("ascii")
                )
            ]
        )
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return tool_err("Cannot view this file: expected UTF-8 text or an image.")
    if "\x00" in text:
        return tool_err("Cannot view this file: expected UTF-8 text or an image.")
    return tool_ok(text)
