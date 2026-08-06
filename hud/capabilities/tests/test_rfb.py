from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import mcp.types as mcp_types
from PIL import Image
from typing_extensions import override

from hud.agents.tools.rfb import RFBTool
from hud.capabilities.rfb import RFBClient


class RecordingRFBTool(RFBTool):
    name = "rfb-test"
    client: Any

    def __init__(self) -> None:
        self.screenshot_mime_type = "image/webp"
        self.client = SimpleNamespace(
            screenshot_png=AsyncMock(return_value=(b"webp", "image/webp")),
        )

    @override
    async def execute(self, arguments: dict[str, Any]) -> Any:
        del arguments
        raise NotImplementedError

    @override
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

    with patch(
        "hud.capabilities.rfb.Image.fromarray", return_value=Image.effect_noise((128, 128), 100)
    ):
        data, mime_type = await client.screenshot_png("image/webp")

    assert mime_type == "image/webp"
    assert data.startswith(b"RIFF")
    assert data[8:12] == b"WEBP"


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
