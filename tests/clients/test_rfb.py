"""``RFBClient`` against the fake VNC screen."""

from __future__ import annotations

import io

from PIL import Image

from hud.capabilities import Capability, RFBClient
from tests.clients.conftest import relayed
from tests.fixtures.screen import fake_screen

COLOR = (10, 200, 30)


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
