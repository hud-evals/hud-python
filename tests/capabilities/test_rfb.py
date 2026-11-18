"""``RFBClient`` against the fake VNC screen."""

from __future__ import annotations

import io

from PIL import Image

from hud.capabilities import Capability, RFBClient
from tests.capabilities.conftest import relayed
from tests.conftest import eventually
from tests.fixtures.screen import fake_screen

COLOR = (10, 200, 30)


async def test_a_screenshot_after_the_connection_drops_reconnects() -> None:
    async with (
        fake_screen(width=4, height=2, color=COLOR, password="s3cret") as screen,
        relayed(Capability.rfb(url=screen.url, password="s3cret")) as (capability, flaky),
    ):
        client = await RFBClient.connect(capability)
        try:
            await client.screenshot()
            flaky.sever()
            data, _ = await client.screenshot()
        finally:
            await client.close()

    assert Image.open(io.BytesIO(data)).convert("RGB").getpixel((0, 0)) == COLOR
    assert flaky.accepted == 2


async def test_each_scroll_direction_turns_the_wheel_an_x_server_reads_for_it() -> None:
    async with fake_screen() as screen:
        client = await RFBClient.connect(Capability.rfb(url=screen.url))
        try:
            client.scroll(2)
            client.scroll(-1)
            client.scroll(1, horizontal=True)
            client.scroll(-1, horizontal=True)
            await client.drain()
            await eventually(lambda: len(screen.pointer_events()) == 10)
        finally:
            await client.close()

    # Mask bit n is X button n + 1: 4 and 5 are the vertical wheel, 6 and 7 the horizontal.
    presses = [event.buttons for event in screen.pointer_events() if event.buttons]
    assert presses == [1 << 4, 1 << 4, 1 << 3, 1 << 6, 1 << 5]
