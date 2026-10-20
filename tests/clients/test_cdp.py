"""``CDPClient`` against the fake DevTools endpoint."""

from __future__ import annotations

import asyncio

import pytest

from hud.capabilities import Capability, CDPClient
from hud.clients import connect
from hud.environment import Environment
from hud.eval import LocalRuntime, Task
from tests.conftest import eventually
from tests.fixtures.browser import fake_browser


@pytest.mark.parametrize(
    ("pages", "target_id", "devtools_path", "connected_to"),
    [
        (["page-1", "page-2"], None, False, "page-1"),
        ([], None, False, "new-1"),
        (["page-1", "page-2"], "page-2", False, "page-2"),
        (["page-1", "page-2"], None, True, "page-2"),
    ],
    ids=["first page", "a new page", "an explicit target", "a devtools url"],
)
async def test_connect_attaches_to_the_chosen_page_target(
    pages: list[str], target_id: str | None, devtools_path: bool, connected_to: str
) -> None:
    async with fake_browser(pages=pages) as browser:
        url = f"{browser.url}/devtools/page/page-2" if devtools_path else browser.url
        client = await CDPClient.connect(Capability.cdp(url=url, target_id=target_id))
        await client.close()

    assert browser.commands
    assert {command.target for command in browser.commands} == {connected_to}


async def test_pending_commands_fail_when_the_browser_closes_the_socket() -> None:
    replies = {"Runtime.evaluate": {"hold": True}, "Browser.crash": {"close": True}}
    async with fake_browser(replies=replies) as browser:
        client = await CDPClient.connect(Capability.cdp(url=browser.url))
        try:
            waiting = asyncio.create_task(client.send("Runtime.evaluate", {"expression": "1"}))
            await eventually(
                lambda: any(command.method == "Runtime.evaluate" for command in browser.commands)
            )
            with pytest.raises(ConnectionError, match="CDP connection closed"):
                await client.send("Browser.crash")
            with pytest.raises(ConnectionError, match="CDP connection closed"):
                await waiting
        finally:
            await client.close()


async def test_a_page_published_at_serve_time_opens_through_the_control_port() -> None:
    async with fake_browser(replies={"Page.navigate": {"result": {"frameId": "f-1"}}}) as browser:
        page = Capability.cdp(name="browser", url=f"{browser.url}/devtools/page/page-1")
        env = Environment("cdp", capabilities=[page])
        async with (
            LocalRuntime(env)(Task(env=env.name, id="scenario")) as runtime,
            connect(runtime) as control,
        ):
            opened = await control.open("browser")
            assert isinstance(opened, CDPClient)
            navigated = await opened.send("Page.navigate", {"url": "https://example.com"})
            again = await control.open("browser")

    assert again is opened
    assert navigated == {"frameId": "f-1"}


async def test_a_cancelled_and_a_timed_out_command_end_and_the_next_one_runs() -> None:
    replies = {"Page.navigate": {"result": {"frameId": "f-1"}}, "Runtime.evaluate": {"hold": True}}
    async with fake_browser(replies=replies) as browser:
        client = await CDPClient.connect(Capability.cdp(url=browser.url))
        try:
            command = asyncio.create_task(client.send("Runtime.evaluate", {"expression": "1"}))
            await eventually(
                lambda: any(sent.method == "Runtime.evaluate" for sent in browser.commands)
            )
            command.cancel()
            with pytest.raises(asyncio.CancelledError):
                await command

            with pytest.raises(TimeoutError):
                await asyncio.wait_for(client.send("Runtime.evaluate", {"expression": "2"}), 0.5)
            await eventually(
                lambda: {"expression": "2"} in [sent.params for sent in browser.commands]
            )

            navigated = await client.send("Page.navigate", {"url": "https://example.com"})
        finally:
            await client.close()

    assert navigated == {"frameId": "f-1"}
