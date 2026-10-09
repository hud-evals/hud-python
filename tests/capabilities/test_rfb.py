from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY, AsyncMock, Mock, patch

import mcp.types as mcp_types
from PIL import Image

from hud.agents.tools.rfb import RFBTool
from hud.capabilities.rfb import RFBClient, WebPScreenshotEncoding


class RecordingRFBTool(RFBTool):
    name = "rfb-test"
    client: Any

    def __init__(self) -> None:
        self.screenshot_encoding = WebPScreenshotEncoding(quality=42)
        self.client = SimpleNamespace(
            screenshot_png=AsyncMock(return_value=(b"webp", "image/webp")),
        )

    async def execute(self, arguments: dict[str, Any]) -> Any:
        del arguments
        raise NotImplementedError

    def to_params(self) -> Any:
        raise NotImplementedError


async def test_screenshot_uses_requested_webp_mime_type() -> None:
    client = object.__new__(RFBClient)
    object.__setattr__(
        client,
        "_conn",
        SimpleNamespace(
            screenshot=AsyncMock(return_value=object()),
        ),
    )

    save = Mock(side_effect=lambda buffer, **_kwargs: buffer.write(b"webp"))
    with patch("hud.capabilities.rfb.Image.fromarray", return_value=SimpleNamespace(save=save)):
        data, mime_type = await client.screenshot_png(WebPScreenshotEncoding(quality=42))

    assert mime_type == "image/webp"
    assert data == b"webp"
    save.assert_called_once_with(ANY, format="WEBP", quality=42)


async def test_screenshot_uses_requested_png_mime_type() -> None:
    client = object.__new__(RFBClient)
    object.__setattr__(
        client,
        "_conn",
        SimpleNamespace(
            screenshot=AsyncMock(return_value=object()),
        ),
    )

    with patch("hud.capabilities.rfb.Image.fromarray", return_value=Image.new("RGB", (8, 8))):
        data, mime_type = await client.screenshot_png("image/png")

    assert mime_type == "image/png"
    assert data.startswith(b"\x89PNG")


async def test_screenshot_reports_encoded_mime_type() -> None:
    tool = RecordingRFBTool()

    result = await tool.screenshot()

    image = result.content[0]
    assert isinstance(image, mcp_types.ImageContent)
    assert image.mimeType == "image/webp"
    tool.client.screenshot_png.assert_awaited_once_with(tool.screenshot_encoding)
