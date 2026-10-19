"""What only ``CDPClient`` does: page-target discovery, error replies, and a closed socket.

The contract every client keeps is in ``test_clients.py``.
"""

from __future__ import annotations

import asyncio

import pytest

from hud.capabilities import Capability, CDPClient
from hud.capabilities.cdp import CDPError
from tests.support import fake_browser

ENABLED = ["Page.enable", "Runtime.enable", "DOM.enable"]


@pytest.mark.parametrize(
    ("pages", "target_id", "devtools_path", "http", "connected_to"),
    [
        (["page-1", "page-2"], None, False, ["GET /json"], "page-1"),
        ([], None, False, ["GET /json", "PUT /json/new?about:blank"], "new-1"),
        (["page-1", "page-2"], "page-2", False, [], "page-2"),
        (["page-1", "page-2"], None, True, [], "page-2"),
    ],
    ids=["first page", "a new page", "an explicit target", "a devtools url"],
)
async def test_connect_attaches_to_a_page_target_and_enables_its_domains(
    pages: list[str],
    target_id: str | None,
    devtools_path: bool,
    http: list[str],
    connected_to: str,
) -> None:
    async with fake_browser(pages=pages) as browser:
        url = f"{browser.url}/devtools/page/page-2" if devtools_path else browser.url
        client = await CDPClient.connect(Capability.cdp(url=url, target_id=target_id))
        await client.close()

    assert browser.http == http
    assert [(command.target, command.method) for command in browser.commands] == [
        (connected_to, method) for method in ENABLED
    ]


async def test_an_error_reply_raises_cdp_error_naming_the_failed_command() -> None:
    error = {"code": -32000, "message": "Cannot navigate to invalid URL"}
    async with fake_browser(replies={"Page.navigate": {"error": error}}) as browser:
        client = await CDPClient.connect(Capability.cdp(url=browser.url))
        try:
            with pytest.raises(CDPError) as raised:
                await client.send("Page.navigate", {"url": "nope"})
        finally:
            await client.close()

    assert (raised.value.code, raised.value.message, str(raised.value)) == (
        -32000,
        "Cannot navigate to invalid URL",
        "CDP 'Page.navigate' failed [-32000]: Cannot navigate to invalid URL",
    )


async def test_pending_commands_fail_when_the_browser_closes_the_socket() -> None:
    replies = {"Runtime.evaluate": {"hold": True}, "Browser.crash": {"close": True}}
    async with fake_browser(replies=replies) as browser:
        client = await CDPClient.connect(Capability.cdp(url=browser.url))
        try:
            waiting = asyncio.create_task(client.send("Runtime.evaluate", {"expression": "1"}))
            await asyncio.sleep(0)
            with pytest.raises(ConnectionError, match="CDP connection closed"):
                await client.send("Browser.crash")
            with pytest.raises(ConnectionError, match="CDP connection closed"):
                await waiting
        finally:
            await client.close()
