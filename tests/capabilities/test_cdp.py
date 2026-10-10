"""``CDPClient`` against the harness DevTools endpoint."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from hud.capabilities import Capability, CDPClient
from hud.capabilities.cdp import CDPError
from tests.harness import Command, fake_browser

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


async def test_connect_fails_when_the_browser_offers_no_page_target() -> None:
    async with fake_browser(pages=[], can_create=False) as browser:
        with pytest.raises(ValueError, match=r"no CDP page target available at 127\.0\.0\.1:"):
            await CDPClient.connect(Capability.cdp(url=browser.url))


async def test_connect_needs_a_host_and_port() -> None:
    with pytest.raises(ValueError, match="cdp capability missing host or port"):
        await CDPClient.connect(Capability(name="browser", protocol="cdp/1.3", url="ws://host"))


@pytest.mark.parametrize(
    ("reply", "result"),
    [({"result": {"frameId": "f-1"}}, {"frameId": "f-1"}), ({}, {})],
    ids=["a result", "an empty reply"],
)
async def test_send_returns_the_command_result(
    reply: dict[str, Any], result: dict[str, Any]
) -> None:
    async with fake_browser(replies={"Page.navigate": reply}) as browser:
        client = await CDPClient.connect(Capability.cdp(url=browser.url))
        try:
            returned = await client.send("Page.navigate", {"url": "https://example.com"})
        finally:
            await client.close()

    assert returned == result
    assert browser.commands[-1] == Command(
        "page-1", "Page.navigate", {"url": "https://example.com"}
    )


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
