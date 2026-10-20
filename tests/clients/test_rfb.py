"""``RFBClient`` against the fake VNC screen."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from hud.capabilities import Capability, RFBClient
from hud.capabilities.rfb import ScreenshotMimeType, WebPScreenshotEncoding
from hud.clients import connect
from hud.environment import Environment
from hud.eval import LocalRuntime, Task
from tests.clients.conftest import relayed
from tests.fixtures.screen import fake_screen

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


async def test_a_screen_published_at_serve_time_opens_through_the_control_port() -> None:
    async with fake_screen(width=4, height=2, color=COLOR) as screen:
        env = Environment("rfb", capabilities=[Capability.rfb(name="screen", url=screen.url)])
        async with (
            LocalRuntime(env)(Task(env=env.name, id="scenario")) as runtime,
            connect(runtime) as control,
        ):
            opened = await control.open("screen")
            assert isinstance(opened, RFBClient)
            data, _ = await opened.screenshot_png()
            again = await control.open("screen")

    assert again is opened
    assert Image.open(io.BytesIO(data)).convert("RGB").getpixel((0, 0)) == COLOR


async def test_a_screenshot_after_the_connection_drops_reconnects() -> None:
    async with (
        fake_screen(width=4, height=2, color=COLOR) as screen,
        relayed(Capability.rfb(url=screen.url)) as (capability, flaky),
    ):
        client = await RFBClient.connect(capability)
        try:
            await client.screenshot_png()
            flaky.sever()
            data, _ = await client.screenshot_png()
        finally:
            await client.close()

    assert Image.open(io.BytesIO(data)).convert("RGB").getpixel((0, 0)) == COLOR
    assert flaky.accepted == 2
