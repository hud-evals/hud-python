"""``RFBClient`` against the harness VNC server."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from hud.capabilities import Capability, RFBClient
from hud.capabilities.rfb import ScreenshotMimeType, WebPScreenshotEncoding
from tests.harness import fake_screen

COLOR = (10, 200, 30)


@pytest.mark.parametrize(
    ("encoding", "mime_type", "image_format"),
    [
        ("image/png", "image/png", "PNG"),
        ("image/webp", "image/webp", "WEBP"),
        (WebPScreenshotEncoding(quality=100), "image/webp", "WEBP"),
    ],
)
async def test_a_screenshot_encodes_the_framebuffer_as_requested(
    encoding: ScreenshotMimeType | WebPScreenshotEncoding, mime_type: str, image_format: str
) -> None:
    async with fake_screen(width=8, height=6, color=COLOR) as screen:
        client = await RFBClient.connect(Capability.rfb(url=screen.url))
        try:
            data, encoded_as = await client.screenshot_png(encoding)
        finally:
            await client.close()

    image = Image.open(io.BytesIO(data))
    assert (encoded_as, image.format, image.size) == (mime_type, image_format, (8, 6))
    colors = image.convert("RGB").getcolors()
    assert colors is not None
    ((count, pixel),) = colors
    assert isinstance(pixel, tuple)
    assert count == 48
    assert max(abs(channel - expected) for channel, expected in zip(pixel, COLOR, strict=True)) <= 4


async def test_a_screenshot_reconnects_when_the_stream_drops() -> None:
    # The first update is the warm-up on connect; the second is dropped.
    async with fake_screen(width=4, height=4, color=COLOR, drop_on_update=2) as screen:
        client = await RFBClient.connect(Capability.rfb(url=screen.url))
        try:
            data, _ = await client.screenshot_png()
        finally:
            await client.close()

    assert screen.connections == 2
    assert Image.open(io.BytesIO(data)).convert("RGB").getpixel((0, 0)) == COLOR


async def test_connecting_needs_a_host_and_port() -> None:
    with pytest.raises(ValueError, match="rfb capability missing host or port"):
        await RFBClient.connect(Capability(name="screen", protocol="rfb/3.8", url="rfb://host"))
